import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

IMAGES = joblib.load('data/original_images')
IMAGES = {_id: cv2.resize(image, (128, 128)) for _id, image in IMAGES.items()}
PERCENTILES = {_id: np.percentile(image, 99) for _id, image in IMAGES.items()}

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

# augmentor = AA.Compose([
#     AA.ShiftScaleRotate(scale_limit=0, rotate_limit=5, shift_limit=0.05, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
#     # AA.GridDistortion(num_steps=3, distort_limit=0.2, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
#     # AA.RandomContrast(limit=0.2, p=1.0),
#     # AA.Blur(blur_limit=3, p=1.0),
#     # AA.OneOf([
#     #     AA.GridDistortion(num_steps=3, distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
#     #     AA.ElasticTransform(alpha=5, sigma=10, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0),
#     # ], p=0.8),
#     # AA.OneOf([
#     #     AA.GaussianBlur(),
#     #     AA.Blur(blur_limit=3),
#     # ], p=0.5),
# ], p=1)

course_dropout1 = AA.CoarseDropout(min_holes=2, max_holes=10, min_height=4, max_height=32, min_width=4, max_width=32, p=1.0)
course_dropout2 = AA.CoarseDropout(min_holes=1, max_holes=2, min_height=32, max_height=64, min_width=32, max_width=64, p=1.0)


def scale_values(image):
    image = image / np.percentile(image, 99)
    return image.clip(0, 1)


def plot_augmentations(random_id=None):

    if not random_id:
        random_id = np.random.choice(list(IMAGES.keys()))

    image = get_image(random_id)
    image = scale_values(image)

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    row, col = 0, 0
    for i in range(16):

        if np.random.rand() <= 0.5:
            aug_img = course_dropout1(image=image)['image']
        else:
            aug_img = course_dropout2(image=image)['image']

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


# plot_augmentations()


class MultiOutputImageGenerator(Sequence):

    def __init__(self, images, batch_size, is_train):
        self.images = images
        self.batch_size = batch_size
        self.is_train = is_train

        tempIds = trainIds[trainIds.index.isin(self.images['image_id'])]

        self.graphemeIds = {}
        for i in range(168):
            self.graphemeIds[i] = list(tempIds[tempIds['grapheme_root'] == i].index)

        self.vowelIds = {}
        for i in range(11):
            self.vowelIds[i] = list(tempIds[tempIds['vowel_diacritic'] == i].index)

        self.consonantIds = {}
        for i in range(7):
            self.consonantIds[i] = list(tempIds[tempIds['consonant_diacritic'] == i].index)

    def __len__(self):
        return int(len(self.images) / self.batch_size)

    def __getitem__(self, idx):

        if self.is_train:
            batchIds = []

            for grapheme_root in [i for i in range(168)]:
                batchIds.append(np.random.choice(self.graphemeIds[grapheme_root]))

            for vowel in [i for i in range(11)]:
                batchIds.append(np.random.choice(self.vowelIds[vowel]))

            for consonant in [i for i in range(7)]:
                batchIds.append(np.random.choice(self.consonantIds[consonant]))

            np.random.shuffle(batchIds)
            batch_images = self.images[self.images['image_id'].isin(batchIds)]
        else:
            batch_images = self.images[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 128, 128, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, row in batch_images.reset_index().iterrows():

            image_id = row['image_id']
            x = IMAGES[image_id].copy()
            x = x / PERCENTILES[image_id]
            x = x.clip(0, 1)
            if self.is_train:
                if np.random.rand() <= 0.5:
                    x = course_dropout1(image=x)['image']
                else:
                    x = course_dropout2(image=x)['image']

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

    def update_weights(self, gr_weights, vc_weights, cd_weights):
        self.images = self.images.drop([
            'grapheme_root_weight', 'vowel_diacritic_weight', 'consonant_diacritic_weight'
        ], axis=1)

        self.images = self.images.merge(gr_weights, on='grapheme_root', how='left')
        self.images = self.images.merge(vc_weights, on='vowel_diacritic', how='left')
        self.images = self.images.merge(cd_weights, on='consonant_diacritic', how='left')

    def make_predictions(self, model):
        grapheme_root_predictions = []
        vowel_diacritic_predictions = []
        consonant_diacritic_predictions = []
        images = []
        for image_id in self.images['image_id']:

            x = IMAGES[image_id].copy()
            x = x / PERCENTILES[image_id]
            x = x.clip(0, 1)
            image = np.stack([x, x, x], axis=2)
            images.append(image)
            if len(images) == 128:
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
            self.images['image_id'].values, grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions
        ], index=['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).T.set_index('image_id')

    def get_output(self, model):
        grapheme_root_predictions = {}
        vowel_diacritic_predictions = {}
        consonant_diacritic_predictions = {}
        for image_id in self.images['image_id']:
            x = get_image(image_id)
            x = scale_values(x)
            image = np.stack([x, x, x], axis=2)
            output = model.predict(np.expand_dims(image, 0))
            grapheme_root_predictions[image_id] = output[0][0]
            vowel_diacritic_predictions[image_id] = output[0][1]
            consonant_diacritic_predictions[image_id] = output[0][2]

        return grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions


def get_data_generators(split, batch_size):

    df = pd.read_csv('data/train.csv')
    splits = pd.read_csv('splits/{}/split.csv'.format(split))
    train_ids = list(splits[splits['split'] == 'train']['image_id'])
    valid_ids = list(splits[splits['split'].isin(['valid', 'test'])]['image_id'])

    train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
    valid_df = df[df['image_id'].isin(valid_ids)].reset_index(drop=True)

    train_generator = MultiOutputImageGenerator(train_df, batch_size, True)
    valid_generator = MultiOutputImageGenerator(valid_df, batch_size, False)
    return train_generator, valid_generator
