import cv2
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from generators import get_data_generators

def train_model(train_generator, valid_generator, backbone_function, connect_head_function, training_path):

    backbone, backbone_output = backbone_function()
    model = connect_head_function(backbone, backbone_output)

    loss = {
    	'grapheme_root': 'categorical_crossentropy',
    	'vowel_diacritic': 'categorical_crossentropy',
        'consonant_diacritic': 'categorical_crossentropy'
    }

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    model.compile(optimizer=Adam(0.001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=300, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/pretrain_history'.format(training_path), 'wb') as f:
        pickle.dump(history.history, f)
    model.save_weights('{}/pretrain_weights.h5'.format(training_path))

    #
    #   FINAL STEP
    #

    for layer in backbone.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=300, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/final_step'.format(training_path), 'wb') as f:
        pickle.dump(history.history, f)
    model.save_weights('{}/final_step_weights.h5'.format(training_path))


####################################
#       PREDICTIONS
###################################

# def make_predictions(image_ids):
#     grapheme_root_predictions = []
#     vowel_diacritic_predictions = []
#     consonant_diacritic_predictions = []
#     images = []
#     for image_id in image_ids:
#         images.append(get_image(image_id))
#         if len(images) == 512:
#             predictions = model.predict(np.array(images))
#             predictions = [p.argmax(axis=1) for p in predictions]
#             grapheme_root_predictions.extend(predictions[0])
#             vowel_diacritic_predictions.extend(predictions[1])
#             consonant_diacritic_predictions.extend(predictions[2])
#             images = []
#     predictions = model.predict(np.array(images))
#     predictions = [p.argmax(axis=1) for p in predictions]
#     grapheme_root_predictions.extend(predictions[0])
#     vowel_diacritic_predictions.extend(predictions[1])
#     consonant_diacritic_predictions.extend(predictions[2])
#     return pd.DataFrame([
#         image_ids, grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions
#     ], index=['image_id', 'grapheme_root', 'consonant_diacritic', 'vowel_diacritic']).T.set_index('image_id')
#
# predictions = make_predictions(valid_generator.ids)
# predictions.head()
#
# trainIds.head()
#
# predictions = predictions.sort_index()
# validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()
# validIds.head(5)
# predictions.columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
#
#
# scores = []
# for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
#     y_true_subset = validIds[component].values.astype(int)
#     y_pred_subset = predictions[component].values.astype(int)
#     scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
# final_score = np.average(scores, weights=[2,1,1])
#
#
# recall_score(validIds['grapheme_root'].astype(int), predictions['grapheme_root'].astype(int), average='macro')
# recall_score(validIds['grapheme_root'].astype(int), predictions['grapheme_root'].astype(int), average='macro')
#
#
# final_score
# final_score
#
# y_true_subset
#
#
# y_pred_subset
