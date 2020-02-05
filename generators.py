import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA
import matplotlib.pyplot as plt

IMAGES = joblib.load('data/images')
trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

augmentor = AA.Compose([
    AA.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, shift_limit=0.05, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.GridDistortion(num_steps=3, distort_limit=0.1, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=0.5)
    # AA.RandomContrast(limit=0.4, p=0.8)
], p=1)

def plot_augmentations():

    random_id = np.random.choice(list(IMAGES.keys()))
    image = get_image(random_id)
    plt.figure(figsize=(5, 5))
    plt.imshow(crop_and_resize_image(image.copy()), cmap='gray')
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    row, col = 0, 0
    for i in range(16):
        aug_img = augmentor(image=image.copy())['image']
        axes[row][col].imshow(crop_and_resize_image(aug_img), cmap='gray')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        if col == 3:
            col = 0
            row += 1
        else:
            col += 1
    plt.tight_layout()
    plt.show()

plot_augmentations()

def get_image(image_id):
    return IMAGES[image_id].copy()


def crop_and_resize_image(image):

    # sum_axis_0 = image.sum(axis=0) > 0
    # sum_axis_1 = image.sum(axis=1) > 0
    # image = image[sum_axis_1, :]
    # image = image[:, sum_axis_0]
    #
    # height, width = image.shape[:2]
    # size = max([width, height])
    #
    # if width < size:
    #     diff = int((size - width) / 2)
    #     image = np.concatenate([np.zeros((height, diff)), image, np.zeros((height, diff))], axis=1)
    # if height < size:
    #     diff = int((size - height) / 2)
    #     image = np.concatenate([np.zeros((diff, width)), image, np.zeros((diff, width))], axis=0)

    image = cv2.resize(image, (64, 64))
    image = image - image.min()
    image = image / image.max()
    # image = image.clip(0, 1)
    return image


class ImageGenerator(Sequence):

    def __init__(self, images, batch_size, is_train):
        self.images = images
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(len(self.images) / self.batch_size)

    def __getitem__(self, idx):

        # if self.is_train:
        #     batch_images = pd.concat([
        #         self.images.sample(n=90, weights='grapheme_root_weight'),
        #         self.images.sample(n=19, weights='vowel_diacritic_weight'),
        #         self.images.sample(n=19, weights='consonant_diacritic_weight')
        #     ])
        # else:
        batch_images = self.images[idx * self.batch_size : (idx+1) * self.batch_size]['image_id']

        X = np.zeros((self.batch_size, 64, 64, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, row in batch_images.reset_index().iterrows():
            image_id = row['image_id']
            x = get_image(image_id)
            # if self.is_train:
            #     x = augmentor(image=x)['image']
            x = crop_and_resize_image(x)
            X[i] = np.stack([x, x, x], axis=2)
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
            self.images = self.images.sample(frac=1)

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

def add_sample_weights(df):

    grapheme_root_counts = df['grapheme_root'].value_counts()
    grapheme_root_counts = grapheme_root_counts.sum() / grapheme_root_counts
    grapheme_root_counts = grapheme_root_counts + len(grapheme_root_counts)
    grapheme_root_counts = grapheme_root_counts.round().reset_index()
    grapheme_root_counts.columns = ['grapheme_root', 'grapheme_root_weight']

    vowel_diacritic_counts = df['vowel_diacritic'].value_counts()
    vowel_diacritic_counts = vowel_diacritic_counts.sum() / vowel_diacritic_counts
    vowel_diacritic_counts = vowel_diacritic_counts + len(vowel_diacritic_counts)
    vowel_diacritic_counts = vowel_diacritic_counts.round().reset_index()
    vowel_diacritic_counts.columns = ['vowel_diacritic', 'vowel_diacritic_weight']

    consonant_diacritic_counts = df['consonant_diacritic'].value_counts()
    consonant_diacritic_counts = consonant_diacritic_counts.sum() / consonant_diacritic_counts
    consonant_diacritic_counts = consonant_diacritic_counts + len(consonant_diacritic_counts)
    consonant_diacritic_counts = consonant_diacritic_counts.round().reset_index()
    consonant_diacritic_counts.columns = ['consonant_diacritic', 'consonant_diacritic_weight']

    df = df.merge(grapheme_root_counts, on='grapheme_root', how='left')
    df = df.merge(vowel_diacritic_counts, on='vowel_diacritic', how='left')
    df = df.merge(consonant_diacritic_counts, on='consonant_diacritic', how='left')

    return df


def get_data_generators(split, batch_size):

    df = pd.read_csv('data/train.csv')
    splits = pd.read_csv('splits/{}/split.csv'.format(split))
    train_ids = list(splits[splits['split'] == 'train']['image_id'])
    valid_ids = list(splits[splits['split'].isin(['valid', 'test'])]['image_id'])

    train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
    valid_df = df[df['image_id'].isin(valid_ids)].reset_index(drop=True)
    train_df = add_sample_weights(train_df)

    train_generator = ImageGenerator(train_df, batch_size, True)
    valid_generator = ImageGenerator(valid_df, batch_size, False)
    return train_generator, valid_generator
