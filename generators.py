import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, pad=16):
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < 236 - 13) else 236
    ymax = ymax + 10 if (ymax < 137 - 10) else 137
    return img0[ymin:ymax,xmin:xmax]


def trim_image(image):

    cut_value = int(image.max() / 2)

    # Remove frame
    frame_indexes_x = (image > 30).sum(axis=1) < 200
    image = image[frame_indexes_x,:]
    frame_indexes_y = (image > 30).sum(axis=0) < 120
    image = image[:,frame_indexes_y]

    image = crop_resize(image)


    for i, (s, c) in enumerate(zip(image.max(axis=0), (image > cut_value).sum(axis=0))):
        if s > cut_value and c > 3:
            image = image[:,i:]
            break
    for i, (s, c) in enumerate(zip(np.flip(image.max(axis=0)), np.flip((image > cut_value).sum(axis=0)))):
        if s > cut_value and c > 3:
            image = image[:,:-i-1]
            break
    for i, (s, c) in enumerate(zip(image.max(axis=1), (image > cut_value).sum(axis=1))):
        if s > cut_value and c > 3:
            image = image[i+1:,:]
            break
    for i, (s, c) in enumerate(zip(np.flip(image.max(axis=1)), np.flip((image > cut_value).sum(axis=1)))):
        if s > cut_value and c > 3:
            image = image[:-i-1,:]
            break

    return image



OIMAGES = joblib.load('data/original_images')
IMAGES = {_id: trim_image(image) for _id, image in OIMAGES.items()}
IMAGES = {_id: cv2.resize(image, (64, 64)) for _id, image in IMAGES.items()}

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

augmentor = AA.Compose([
    # AA.ShiftScaleRotate(scale_limit=0.1, rotate_limit=5, shift_limit=0.1, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.GridDistortion(num_steps=3, distort_limit=0.2, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.RandomContrast(limit=0.2, p=1.0),
    # AA.Blur(blur_limit=3, p=1.0),
    # GridMask(num_grid=(3, 7), rotate=10, p=1.0),
    AA.CoarseDropout(min_holes=1, max_holes=10, min_height=4, max_height=8, min_width=4, max_width=8, p=0.8)
], p=1)



def plot_preprocessing(img_id):
    print(img_id)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
    axes[0].imshow(OIMAGES[img_id], cmap='gray')
    axes[1].imshow(trim_image(OIMAGES[img_id]), cmap='gray')
    axes[2].imshow(get_trimmed_image(img_id), cmap='gray')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.tight_layout()
    plt.show()


# O['Train_131498'].max() / 2
#
# O['Train_41126'].max(axis=1)
#
#
# plt.imshow(O['Train_41126'], cmap='gray')
#
# plt.imshow(trim_image(O['Train_41126']), cmap='gray')


def get_original_image(image_id):
    x = OIMAGES[image_id].copy()
    x = cv2.resize(x, (64, 64))
    x = x / np.percentile(x, 99.9)
    return x.clip(0, 1)

def get_trimmed_image(image_id):
    x = IMAGES[image_id].copy()
    x = x / np.percentile(x, 99)
    return x.clip(0, 1)


def plot_augmentations():

    _id = np.random.choice(list(IMAGES.keys()))
    print(_id)
    image = get_image(_id)

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    row, col = 0, 0
    for i in range(16):

        aug_img = augmentor(image=image.copy())['image']
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

        batch_images = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        X = np.zeros((self.batch_size, 64, 64, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, row in batch_images.reset_index().iterrows():

            ox = get_original_image(row['image_id'])
            px = get_trimmed_image(row['image_id'])

            # if self.is_train:
            #     x = augmentor(image=x)['image']

            X[i] = np.stack([ox, px, np.zeros((64, 64))], axis=2)
            grapheme_root_Y[i][trainIds.loc[row['image_id']]['grapheme_root']] = 1
            vowel_diacritic_Y[i][trainIds.loc[row['image_id']]['vowel_diacritic']] = 1
            consonant_diacritic_Y[i][trainIds.loc[row['image_id']]['consonant_diacritic']] = 1

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
            x = get_image(image_id)
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
