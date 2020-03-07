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
from utils.weighted_recall import WeightedRecall
from utils.binary_focal_loss import binary_focal_loss


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


def pretrain_model(model, name, settings):

    if not os.path.exists('results/{}'.format(name)):
        os.makedirs('results/{}'.format(name))

    train_generator, valid_generator = get_data_generators(settings['split'], settings['batchsize'])
    weighted_recall = WeightedRecall(train_generator, valid_generator)
    model.compile(optimizer=Adam(settings['learning_rate']), loss='binary_crossentropy')
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

    print('Preparing Callbacks...')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('results/{}/train_full.h5'.format(name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    if recall:
        weighted_recall = WeightedRecall(train_generator, valid_generator)
        callbacks = [weighted_recall, reduce_lr, early_stopping, model_checkpoint]
    else:
        callbacks = [reduce_lr, early_stopping, model_checkpoint]

    print('Training Model...')
    # loss=[binary_focal_loss(alpha=.25, gamma=2)]
    model.compile(optimizer=Adam(settings['learning_rate']), loss=[binary_focal_loss(alpha=.25, gamma=1)], metrics=['accuracy', 'binary_accuracy'])
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

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['categorical_accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_generator.__len__(), epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[weighted_recall, early_stopping, reduce_lr]
    )

    with open('results/{}/head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('results/{}/model.h5'.format(training_path))
