####################################
#       IMPORTS
###################################

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as AA
import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence

####################################
#       DATA GENERATOR
###################################

IMAGES = joblib.load('data/images')
trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

def get_image(image_id):

    x = IMAGES[image_id]
    x1 = x >= 80
    x2 = x >= 150
    x = x - x.min()
    x = x / x.max()
    return np.stack([x, x1, x2], axis=2)

class ImageGenerator(Sequence):

    def __init__(self, ids, batch_size, is_train):
        self.ids = ids
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):

        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 128, 128, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, image_id in enumerate(batch_ids):
            X[i] = get_image(image_id)
            grapheme_root_Y[i][trainIds.loc[image_id]['grapheme_root']] = 1
            vowel_diacritic_Y[i][trainIds.loc[image_id]['vowel_diacritic']] = 1
            consonant_diacritic_Y[i][trainIds.loc[image_id]['consonant_diacritic']] = 1

        return X, {
            'grapheme_root': grapheme_root_Y,
            'vowel_diacritic': vowel_diacritic_Y,
            'consonant_diacritic': consonant_diacritic_Y
        }

    def on_epoch_end(self):
        if self.is_train:
            np.random.shuffle(self.ids)

splits = pd.read_csv('splits/split1/split.csv')
train_generator = ImageGenerator(list(splits[splits['split'] == 'train']['image_id']), 64, True)
valid_generator = ImageGenerator(list(splits[splits['split'] == 'valid']['image_id']), 128, True)

####################################
#       ALGORITHM
###################################

# Backbone
backbone = efn.EfficientNetB0(input_shape=(128, 128, 3), include_top=False,  weights='imagenet')
global_average = GlobalAveragePooling2D()(backbone.output)
for layer in backbone.layers:
    layer.trainable = False

# grapheme_root_head
grapheme_root_dense = Dense(64, activation='relu', name='grapheme_root_dense')(global_average)
grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(grapheme_root_dense)

# vowel_diacritic_head
vowel_diacritic_dense = Dense(32, activation='relu', name='vowel_diacritic_dense')(global_average)
vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(vowel_diacritic_dense)

# consonant_diacritic_head
consonant_diacritic_dense = Dense(32, activation='relu', name='consonant_diacritic_dense')(global_average)
consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(consonant_diacritic_dense)

# Model
model = Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])


model.summary()

####################################
#       TRAINING
###################################

valid_generator.__len__()

losses = {
	'grapheme_root': 'categorical_crossentropy',
	'vowel_diacritic': 'categorical_crossentropy',
    'consonant_diacritic': 'categorical_crossentropy'
}

model.compile(optimizer='adam', loss=losses, metrics=['categorical_accuracy'])

history = model.fit_generator(
    train_generator, steps_per_epoch=10, epochs=5,
    validation_data=valid_generator, validation_steps=valid_generator.__len__(),
)
