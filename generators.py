import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA

IMAGES = joblib.load('data/images')
trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

augmentor = AA.Compose([
    AA.ShiftScaleRotate(scale_limit=0.08, rotate_limit=10, shift_limit=0.08, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
    AA.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.8),
    AA.GridDistortion(num_steps=5, distort_limit=0.2, p=0.8),
    AA.RandomContrast(limit=0.5, always_apply=True)
], p=1)


import matplotlib.pyplot as plt

def plot_augmentations():

    random_id = np.random.choice(list(IMAGES.keys()))
    plt.figure(figsize=(5, 5))
    plt.imshow(IMAGES[random_id], cmap='gray')
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    row, col = 0, 0
    for i in range(16):
        aug_img = augmentor(image=IMAGES[random_id].copy())['image']
        axes[row][col].imshow(aug_img, cmap='gray')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        if col == 3:
            col = 0
            row += 1
        else:
            col += 1
    plt.tight_layout()
    plt.show()

def get_image(image_id):

    x = IMAGES[image_id]
    x = x - x.min()
    x = x / x.max()
    return np.expand_dims(x, 2)

class ImageGenerator(Sequence):

    def __init__(self, ids, batch_size, is_train):
        self.ids = ids
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):

        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 128, 128, 1))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, image_id in enumerate(batch_ids):
            x = get_image(image_id)
            if self.is_train:
                x = augmentor(image=x)['image']
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

    def make_predictions(self, model):
        grapheme_root_predictions = []
        vowel_diacritic_predictions = []
        consonant_diacritic_predictions = []
        images = []
        for image_id in self.ids:
            images.append(get_image(image_id))
            if len(images) == 512:
                predictions = model.predict(np.array(images))
                predictions = [p.argmax(axis=1) for p in predictions]
                grapheme_root_predictions.extend(predictions[0])
                vowel_diacritic_predictions.extend(predictions[1])
                consonant_diacritic_predictions.extend(predictions[2])
                images = []
        predictions = model.predict(np.array(images))
        predictions = [p.argmax(axis=1) for p in predictions]
        grapheme_root_predictions.extend(predictions[0])
        vowel_diacritic_predictions.extend(predictions[1])
        consonant_diacritic_predictions.extend(predictions[2])
        return pd.DataFrame([
            self.ids, grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions
        ], index=['image_id', 'grapheme_root', 'consonant_diacritic', 'vowel_diacritic']).T.set_index('image_id')

def get_data_generators(split, batch_size):
    splits = pd.read_csv('splits/{}/split.csv'.format(split))
    train_generator = ImageGenerator(list(splits[splits['split'] == 'train']['image_id']), 128, True)
    valid_generator = ImageGenerator(list(splits[splits['split'] == 'valid']['image_id']), 128, False)
    return train_generator, valid_generator
