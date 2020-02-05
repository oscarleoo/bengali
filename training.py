import cv2
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.metrics import recall_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from generators import get_data_generators

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

def calculate_class_weights(y_true, y_pred, title, alpha):
    MCM = multilabel_confusion_matrix(
        y_true[title].values.astype(int),
        y_pred[title].values.astype(int)
    )
    true_positives = MCM[:, 1, 1]
    true_sum = tp_sum + MCM[:, 1, 0]
    class_recall = (true_positives / true_sum) + alpha
    class_recall = 1 / class_recall
    class_recall = class_recall / class_recall.sum()
    class_index = [i for i in range(len(class_recall))]
    return pd.DataFrame([class_index, class_recall], index=[title, '{}_weight'.format(title)]).T


def calculate_recall(y_true, y_pred):
    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = y_true[component].values.astype(int)
        y_pred_subset = predictions[component].values.astype(int)
        scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
    return round(np.average(scores, weights=[2,1,1]), 5)


class WeightedRecall(Callback):

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def on_epoch_end(self, batch, logs={}):

        print(self.train.images.head())

        predictions = self.valid.make_predictions(self.model).sort_index()
        validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()
        score = calculate_recall(validIds, predictions)
        print('==> Weighted Recal Score:', score)

        gr_weights = calculate_class_weights(validIds, predictions, 'grapheme_root', 0.1)
        vc_weights = calculate_class_weights(validIds, predictions, 'vowel_diacritic', 0.1)
        cd_weights = calculate_class_weights(validIds, predictions, 'consonant_diacritic', 0.1)
        self.train.update_weights(gr_weights, vc_weights, cd_weights)
        return


def train_model(train_generator, valid_generator, backbone_function, connect_head_function, training_path, title):

    backbone, backbone_output = backbone_function()
    model = connect_head_function(backbone, backbone_output)
    weighted_recall = WeightedRecall(train_generator, valid_generator)

    loss = {
    	'grapheme_root': 'categorical_crossentropy',
    	'vowel_diacritic': 'categorical_crossentropy',
        'consonant_diacritic': 'categorical_crossentropy'
    }

    #loss_weights = {'grapheme_root': 0.685, 'vowel_diacritic': 0.175, 'consonant_diacritic': 0.14}
    # loss_weights = {'grapheme_root': 1, 'vowel_diacritic': 1, 'consonant_diacritic': 1}


    ###############################
    #   PRETRAINING
    ###############################

    print()
    print('Pretraining full network with lr=0.001...')
    print()

    model.compile(optimizer=Adam(0.001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=5,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall]
    )
    with open('{}/{}_pretrain_history'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    ###############################
    #   TRAINING
    ###############################

    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=5, restore_best_weights=True, verbose=1)

    print()
    print('Training full algorithm with early stoppping and decay')
    print()

    model.compile(optimizer=Adam(0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, reduce_lr, early_stopping]
    )

    with open('{}/{}_full_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    #
    #   FINAL STEP
    #

    for layer in backbone.layers:
        layer.trainable = False

    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=3, restore_best_weights=True, verbose=1)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, early_stopping]
    )

    with open('{}/{}_head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('{}/model.h5'.format(training_path))
