import os
import cv2
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model
from sklearn.metrics import recall_score, multilabel_confusion_matrix
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from generators import get_data_generators
from utils.weighted_recall import WeightedRecall

def get_loss():

    return {
    	'grapheme_root': 'categorical_crossentropy',
    	'vowel_diacritic': 'categorical_crossentropy',
        'consonant_diacritic': 'categorical_crossentropy'
    }, {
        'grapheme_root': 1,
        'vowel_diacritic': 1,
        'consonant_diacritic': 1
    }


def pretrain_model(model, name, settings):

    if not os.path.exists('results/{}'.format(name)):
        os.makedirs('results/{}'.format(name))

    loss, loss_weights = get_loss()
    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    model.compile(optimizer=Adam(settings['learning_rate']), loss=loss, loss_weights=loss_weights, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=settings['steps_per_epoch'], epochs=settings['epochs'],
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall]
    )
    with open('results/{}/pretrain_history'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('results/{}/pretrain_model.h5'.format(name))


def train_full_model(split, name, settings):

    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    model = load_model('results/{}/pretrain_model.h5'.format(name))
    loss, loss_weights = get_loss()

    weighted_recall = WeightedRecall(train_generator, valid_generator)
    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=3, restore_best_weights=True, verbose=1)

    model.compile(optimizer=Adam(0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=5000, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, reduce_lr, early_stopping]
    )

    with open('results/{}/full_train'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('results/{}/train_full.h5'.format(name))


def train_head(model, backend, split, name, settings):

    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    loss, loss_weights = get_loss()

    for layer in backbone.layers:
        layer.trainable = False

    weighted_recall = WeightedRecall(train_generator, valid_generator)
    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=3, restore_best_weights=True, verbose=1)

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=5000, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, early_stopping, reduce_lr]
    )

    with open('results/{}/head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('results/{}/model.h5'.format(training_path))
