import cv2
import joblib
import numpy as np
import pandas as pd
from keras.utils import Sequence
import albumentations as AA
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

def black_threshold(img):
    img = img[1:-1, 1:-1]
    return img * (img > (img.max() / 5))


def get_component_shape(component):

    component = component[3:-3, 3:-3]
    if component.max() == 0:
        return (1, 1)

    x_filter = component.max(axis=1) > 0
    component = component[x_filter, :]

    y_filter = component.max(axis=0) > 0
    component = component[:, y_filter]

    return component.shape


def get_coordinates(component):

    contours = cv2.findContours(component.astype(np.uint8).clip(0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extreme_points = contours[1][0].squeeze()

    try:
        x_min, x_max = extreme_points[:,0].min(), extreme_points[:,0].max()
        y_min, y_max = extreme_points[:,1].min(), extreme_points[:,1].max()
        return x_min, x_max, y_min, y_max
    except:
        return 0, 0, 0, 0


def filter_on_distance(components):

    extreme_points = [c[3] for c in components]
    reference = np.argmax([(p[1] - p[0]) * (p[3] - p[2]) for p in extreme_points])
    reference = extreme_points[reference]
    distances = [max([
        points[0] - reference[1],
        reference[0] - points[1],
        points[2] - reference[3],
        reference[2] - points[3]
    ]) for points in extreme_points]

    return [c for c, d in zip(components, distances) if d < 40]


def remove_unwanted_components(img):

    img_max = img.max()
    new_image = np.zeros(img.shape)
    num_component, component = cv2.connectedComponents(img)

    components = [(
        component == c,
        get_component_shape(component == c),
        ((component == c) * img).max(),
        get_coordinates(component == c)
    ) for c in range(1, num_component)]

    components = filter_on_distance(components)

    for p, shape, p_max, contours in components:
        min_shape, max_shape = min(shape), max(shape)
        shape_ratio = max_shape / min_shape

        if p.sum() > 60 and min_shape >= 5 and p_max >= (img_max / 2) and shape_ratio <= 12:
            new_image += (img * p)

    return new_image.clip(0, 255)


def pad_image(img):

    height, width = img.shape
    if height > width:
        diff = int((height - width) / 2)
        padding = np.zeros((height, diff))
        img = np.concatenate([padding, img, padding], axis=1)
    elif width > height:
        diff = int((width - height) / 2)
        padding = np.zeros((diff, width))
        img = np.concatenate([padding, img, padding], axis=0)

    return img


def trim_image(img):

    trim_value = img.max() / 5
    height, width = img.shape
    for i, (s, c) in enumerate(zip(img.max(axis=0), (img > 5).sum(axis=0))):
        if s > trim_value and c > 3:
            if i > 5:
                img = img[:,i-5:]
            else:
                img = np.concatenate([np.zeros((height, 5-i)), img], axis=1)
            break

    height, width = img.shape
    for i, (s, c) in enumerate(zip(np.flip(img.max(axis=0)), np.flip((img > 5).sum(axis=0)))):
        if s > trim_value and c > 3:
            if i > 5:
                img = img[:,:-i + 5]
            else:
                img = np.concatenate([img, np.zeros((height, 5-i))], axis=1)
            break

    height, width = img.shape
    for i, (s, c) in enumerate(zip(img.max(axis=1), (img > 5).sum(axis=1))):
        if s > trim_value and c > 3:
            if i > 5:
                img = img[i-5:,:]
            else:
                img = np.concatenate([np.zeros((5-i, width)), img], axis=0)
            break

    height, width = img.shape
    for i, (s, c) in enumerate(zip(np.flip(img.max(axis=1)), np.flip((img > 5).sum(axis=1)))):
        if s > trim_value and c > 3:
            if i > 5:
                img = img[:-i + 5,:]
            else:
                img = np.concatenate([img, np.zeros((5-i, width))], axis=0)
            break

    return pad_image(img)


def preprocess_original_image(img):
    img = black_threshold(img)
    img = remove_unwanted_components(img)
    img = trim_image(img)
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)


IMAGES = joblib.load('data/original_images')
IMAGES = {_id: preprocess_original_image(image) for _id, image in IMAGES.items()}


trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

augmentor = AA.Compose([
    AA.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit=0.1, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    AA.CoarseDropout(min_holes=1, max_holes=10, min_height=4, max_height=12, min_width=4, max_width=12, p=1.0)
], p=1)


def get_image(image_id):
    x = IMAGES[image_id].copy()
    x = x / np.percentile(x, 98)
    return x.clip(0, 1)


def plot_augmentations():

    _id = np.random.choice(list(IMAGES.keys()))
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


        # if self.is_train:
        #     batchIds = []
        #
        #     for grapheme_root in [i for i in range(168)]:
        #         batchIds.append(np.random.choice(self.graphemeIds[grapheme_root]))
        #
        #     for vowel in [i for i in range(11)]:
        #         batchIds.append(np.random.choice(self.vowelIds[vowel]))
        #
        #     for consonant in [i for i in range(7)]:
        #         batchIds.append(np.random.choice(self.consonantIds[consonant]))
        #
        #     np.random.shuffle(batchIds)
        #     batch_images = self.images[self.images['image_id'].isin(batchIds)]
        # else:
        batch_images = self.images[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 96, 96, 3))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i, row in batch_images.reset_index().iterrows():

            x = get_image(row['image_id'])

            if self.is_train:
                x = augmentor(image=x)['image']

            X[i] = np.stack([x, x, x], axis=2)

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
