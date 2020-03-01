import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

IMAGES = joblib.load('data/images')
trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

augmentor = AA.Compose([
    AA.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.GridDistortion(num_steps=3, distort_limit=0.2, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.RandomContrast(limit=0.2, p=1.0),
    # AA.Blur(blur_limit=3, p=1.0),
    # AA.OneOf([
    #     AA.GridDistortion(num_steps=5, distort_limit=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
    #     AA.ElasticTransform(alpha=5, sigma=30, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0),
    # ], p=0.9),
    # AA.OneOf([
    #     AA.GaussianBlur(),
    #     AA.Blur(blur_limit=3),
    # ], p=0.5),
], p=1)

course_dropout = AA.CoarseDropout(min_holes=2, max_holes=10, min_height=6, max_height=16, min_width=6, max_width=16, p=1.0)

def get_image(image_id):
    return IMAGES[image_id].copy()


def trim_image(image):

    sum_axis_0 = image.sum(axis=0) > 0.1
    sum_axis_1 = image.sum(axis=1) > 0.1
    image = image[sum_axis_1, :]
    image = image[:, sum_axis_0]

    return image


def pad_image(image, train=False):

    height, width = image.shape[:2]
    size = max([width, height])

    if width < size:
        missing = size - width
        if train:
            diff = np.random.randint(missing)
        else:
            diff = int(missing / 2)
        image = np.concatenate([np.zeros((height, diff)), image, np.zeros((height, missing - diff))], axis=1)
    if height < size:
        missing = size - height
        if train:
            diff = np.random.randint(missing)
        else:
            diff = int(missing / 2)
        image = np.concatenate([np.zeros((diff, width)), image, np.zeros((missing - diff, width))], axis=0)

    return cv2.resize(image, (64, 64))


def scale_values(image):

    image = image - image.min()
    image = image / np.percentile(image, 99)
    return image.clip(0, 1)


def get_cut_values(image):
    height, width = image.shape[:2]
    for l0, s in enumerate(image.sum(axis=0)):
        if s >= 5: break
    for r0, s in enumerate(np.flip(image.sum(axis=0))):
        if s >= 5: break
    for l1, s in enumerate(image.sum(axis=1)):
        if s >= 5: break
    for r1, s in enumerate(np.flip(image.sum(axis=1))):
        if s >= 5: break

    return l0, r0, l1, r1


def random_trim(image, l0, r0, l1, r1):

    height, width = image.shape[:2]

    p = np.array([l0 + 2, r0 + 2, l1 + 2, r1 + 2]) / (l0 + r0 + l1 + r1 + 8)
    r = np.random.choice(['l0', 'r0', 'l1', 'r1'], p=p)

    if r == 'l0':
        image = image[np.random.randint(l1 + 1):,:]
    elif r == 'r0':
        image = image[:np.random.randint(height - r1, height + 1),:]
    elif r == 'l1':
        image = image[:,np.random.randint(l0 + 1):]
    elif r == 'r1':
        image = image[:,:np.random.randint(width - r0, width + 1)]

    return image


def plot_augmentations(random_id=None):

    if not random_id:
        random_id = np.random.choice(list(IMAGES.keys()))

    image = get_image(random_id)
    image = scale_values(image)

    plt.figure(figsize=(5, 5))
    plt.imshow(pad_image(trim_image(image)), cmap='gray')
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    row, col = 0, 0
    for i in range(16):
        aug_img = augmentor(image=image.copy())['image']
        aug_img = trim_image(aug_img)
        l0, r0, l1, r1 = get_cut_values(aug_img)
        aug_img = random_trim(aug_img.copy(), l0, r0, l1, r1 )
        aug_img = pad_image(aug_img)
        aug_img = course_dropout(image=aug_img)['image']
        axes[row][col].imshow(pad_image(aug_img), cmap='gray')
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

        X = np.zeros((self.batch_size, 64, 64, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, row in batch_images.reset_index().iterrows():

            image_id = row['image_id']
            x = get_image(image_id)
            x = scale_values(x)
            if self.is_train:
                x = augmentor(image=x)['image']
                x = trim_image(x)
                l0, r0, l1, r1 = get_cut_values(x)
                x = random_trim(x, l0, r0, l1, r1)
                x = pad_image(x, self.is_train)
                x = course_dropout(image=x)['image']
            else:
                x = trim_image(x)
                x = pad_image(x, self.is_train)

            plt.tight_layout()
            plt.show()
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
            image = get_image(image_id)
            image = scale_values(image)
            image = trim_image(image)
            image = pad_image(image, False)
            image = np.stack([image, image, image], axis=2)
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
