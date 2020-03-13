import os
import cv2
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import recall_score, multilabel_confusion_matrix
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint

from generators import get_data_generators
from utils.weighted_recall import WeightedRecall, calculate_recall
from utils.focal_loss import categorical_focal_loss


def swish(x, beta = 1):
    return (x * K.sigmoid(beta * x))


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


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


def test_performance(model, name):

    model.load_weights('results/{}/train_full.h5'.format(name))
    train_generator, valid_generator = get_data_generators(name, 128)

    valid_predictions = valid_generator.make_predictions(model).sort_index()
    train_predictions = train_generator.make_predictions(model).sort_index()

    valid_score, valid_gr_score, valid_vd_score, valid_cd_score = calculate_recall(valid_predictions)
    train_score, train_gr_score, train_vd_score, train_cd_score = calculate_recall(train_predictions)

    print('==> Weighted Valid Recal Score: {} ({} - {} - {})'.format(valid_score, valid_gr_score, valid_vd_score, valid_cd_score))
    print('==> Weighted Train Recal Score: {} ({} - {} - {})'.format(train_score, train_gr_score, train_vd_score, train_cd_score))


def pretrain_model(model, name, settings):

    if not os.path.exists('results/{}'.format(name)):
        os.makedirs('results/{}'.format(name))

    loss, loss_weights = get_loss()
    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    weighted_recall = WeightedRecall(train_generator, valid_generator)
    model.compile(optimizer=Adam(settings['learning_rate']), loss=loss, loss_weights=loss_weights)
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_generator.__len__(), epochs=settings['epochs'],
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
    )
    with open('results/{}/pretrain_history'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save_weights('results/{}/pretrain_model.h5'.format(name))


def train_full_model(model, name, settings, retrain=False, recall=False):

    print('Getting Generators...')
    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    print('Loading Model...')
    if retrain:
        model.load_weights('results/{}/train_full.h5'.format(name))
    else:
        model.load_weights('results/{}/pretrain_model.h5'.format(name))
    loss, loss_weights = get_loss()

    print('Preparing Callbacks...')
    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.2, patience=3, min_lr=0.0000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=5, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('results/{}/train_full.h5'.format(name), monitor='val_grapheme_root_loss', verbose=1, save_best_only=True, save_weights_only=True)
    if recall:
        weighted_recall = WeightedRecall(train_generator, valid_generator)
        callbacks = [weighted_recall, reduce_lr, early_stopping, model_checkpoint]
    else:
        callbacks = [reduce_lr, early_stopping, model_checkpoint]

    print('Training Model...')
    model.compile(optimizer=Adam(settings['learning_rate'], clipnorm=0.1), metrics=['categorical_accuracy'], loss=loss, loss_weights=loss_weights)
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_generator.__len__(), epochs=settings['epochs'],
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=callbacks
    )

    print('Saving Model...')
    with open('results/{}/full_train'.format(name), 'wb') as f:
        pickle.dump(history.history, f)

    model.save_weights('results/{}/train_full.h5'.format(name))
    print('Done')


def train_head(model, backend, split, name, settings):

    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    loss, loss_weights = get_loss()

    for layer in backbone.layers:
        layer.trainable = False

    weighted_recall = WeightedRecall(train_generator, valid_generator)
    reduce_lr = ReduceLROnPlateau(monitor='val_grapheme_root_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_grapheme_root_loss', patience=3, restore_best_weights=True, verbose=1)

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'], clipnorm=0.1)
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_generator.__len__(), epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, early_stopping, reduce_lr]
    )

    with open('results/{}/head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('results/{}/model.h5'.format(training_path))
