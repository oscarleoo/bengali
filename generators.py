import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence

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

    def __init__(self, ids, batch_size, is_train, augmentor=None):
        self.ids = ids
        self.batch_size = batch_size
        self.is_train = is_train
        self.augmentor = augmentor

    def __len__(self):
        return int(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):

        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 128, 128, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, image_id in enumerate(batch_ids):
            x = get_image(image_id)
            if self.is_train:
                x = self.augmentor(image=x)['image']
            X[i] = x
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
            ids = [id_ for id_ in self.ids]
            np.random.shuffle(ids)
            self.ids = ids

def get_data_generators(split, augmentor, batch_size):
    splits = pd.read_csv('splits/{}/split.csv'.format(split))
    train_generator = ImageGenerator(list(splits[splits['split'] == 'train']['image_id']), 128, True, augmentor)
    valid_generator = ImageGenerator(list(splits[splits['split'] == 'valid']['image_id']), 128, False)
    return train_generator, valid_generator
