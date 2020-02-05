import cv2
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from generators import get_data_generators

import keras.backend as K

# hmm1 = K.random_normal((10, 5))
# hmm2 = K.zeros((10, 5))
#
# hmm1
# K.max(hmm1, axis=0)
# K.stack([K.max(hmm1, axis=0) for i in range(10)])
# hmm2.shape
# K.flatten(K.equal(hmm1, K.stack([K.max(hmm1, axis=0) for i in range(10)])))

def weighted_recall(y_true, y_pred):

    gr_pred = K.cast(K.flatten(K.equal(y_pred[:, :168], K.stack([K.max(y_pred[:, :168], axis=0) for i in range(64)]))), dtype='float32')
    gr_true = K.flatten(y_true[:, :168])

    return K.sum(gr_pred * gr_true) / K.sum(gr_true)


def train_model(train_generator, valid_generator, backbone_function, connect_head_function, training_path, title):

    model, backbone = backbone_function()
    # model = connect_head_function(backbone, backbone_output)

    # loss = {
    # 	'grapheme_root': 'categorical_crossentropy',
    # 	'vowel_diacritic': 'categorical_crossentropy',
    #     'consonant_diacritic': 'categorical_crossentropy'
    # }

    #loss_weights = {'grapheme_root': 0.685, 'vowel_diacritic': 0.175, 'consonant_diacritic': 0.14}
    # loss_weights = {'grapheme_root': 1, 'vowel_diacritic': 1, 'consonant_diacritic': 1}


    ###############################
    #   PRETRAINING
    ###############################

    print()
    print('Pretraining full network with lr=0.000001 and lr=0.001...')
    print()

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=[weighted_recall])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=5,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
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

    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/{}_full_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    #
    #   FINAL STEP
    #

    for layer in backbone.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=500, epochs=1000,
        validation_data=valid_generator, validation_steps=valid_generator.__len__(),
        callbacks=[reduce_lr, early_stopping]
    )

    with open('{}/{}_head_train'.format(training_path, title), 'wb') as f:
        pickle.dump(history.history, f)

    model.save('{}/model.h5'.format(training_path))
