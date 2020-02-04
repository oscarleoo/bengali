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

    loss_weights = {'grapheme_root': 0.685, 'vowel_diacritic': 0.175, 'consonant_diacritic': 0.14}

    for layer in backbone.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(0.0001), loss=loss, loss_weights=loss_weights, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=3,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
    )

    for layer in backbone.layers:
        layer.trainable = True

    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=5, restore_best_weights=True, verbose=1)

    model.compile(optimizer=Adam(0.0001), loss=loss, loss_weights=loss_weights, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/pretrain_history'.format(training_path), 'wb') as f:
        pickle.dump(history.history, f)

    #
    #   FINAL STEP
    #

    for layer in backbone.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, loss_weights=loss_weights, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/final_step'.format(training_path), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('{}/model.h5')
