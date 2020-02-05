import cv2
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.metrics import recall_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from generators import get_data_generators

class WeightedRecall(Callback):

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, batch, logs={}):
        score = self.generator.recall(self.model)
        print(score)
        return


def train_model(train_generator, valid_generator, backbone_function, connect_head_function, training_path, title):

    backbone, backbone_output = backbone_function()
    model = connect_head_function(backbone, backbone_output)
    weighted_recall = WeightedRecall(valid_generator)

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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, early_stopping]
    )

    with open('{}/{}_head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('{}/model.h5'.format(training_path))
