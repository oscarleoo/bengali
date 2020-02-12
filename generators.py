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
    AA.ShiftScaleRotate(scale_limit=0, rotate_limit=5, shift_limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.GridDistortion(num_steps=3, distort_limit=0.2, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    # AA.RandomContrast(limit=0.2, p=1.0),
    # AA.Blur(blur_limit=3, p=1.0),
    AA.OneOf([
        AA.GridDistortion(num_steps=3, distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
        AA.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], p=1.0),
    AA.OneOf([
        AA.GaussianBlur(),
        AA.Blur(blur_limit=3),
    ], p=0.5),
    # AA.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.8),
], p=1)

# %timeit x=augmentor(image=image.copy())['image']

# plot_augmentations('Train_12451')

def plot_augmentations(random_id=None):
    if not random_id:
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


def get_image(image_id):
    return IMAGES[image_id].copy()


def crop_and_resize_image(image):

    sum_axis_0 = image.sum(axis=0) > 100
    sum_axis_1 = image.sum(axis=1) > 100
    image = image[sum_axis_1, :]
    image = image[:, sum_axis_0]

    height, width = image.shape[:2]
    size = max([width, height])

    if width < size:
        diff = int((size - width) / 2)
        image = np.concatenate([np.zeros((height, diff)), image, np.zeros((height, diff))], axis=1)
    if height < size:
        diff = int((size - height) / 2)
        image = np.concatenate([np.zeros((diff, width)), image, np.zeros((diff, width))], axis=0)

    image = cv2.resize(image, (64, 64))
    image = image - image.min()
    image = image / np.percentile(image, 99)
    image = image.clip(0, 1)
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
        Y = np.zeros((self.batch_size, 168 + 11 + 7))

        for i, row in batch_images.reset_index().iterrows():
            image_id = row['image_id']
            x = get_image(image_id)
            if self.is_train:
                x = augmentor(image=x)['image']
            x = crop_and_resize_image(x)
            X[i] = np.stack([x, x, x], axis=2)
            grapheme_root_Y = [0 for i in range(168)]
            vowel_diacritic_Y = [0 for i in range(11)]
            consonant_diacritic_Y = [0 for i in range(7)]
            grapheme_root_Y[trainIds.loc[image_id]['grapheme_root']] = 1
            vowel_diacritic_Y[trainIds.loc[image_id]['vowel_diacritic']] = 1
            consonant_diacritic_Y[trainIds.loc[image_id]['consonant_diacritic']] = 1
            Y[i] = np.array(grapheme_root_Y + vowel_diacritic_Y + consonant_diacritic_Y)

        return X, Y

    def make_predictions(self, model):
        grapheme_root_predictions = []
        vowel_diacritic_predictions = []
        consonant_diacritic_predictions = []
        images = []
        for image_id in self.images['image_id']:
            image = crop_and_resize_image(get_image(image_id))
            image = np.stack([image, image, image], axis=2)
            images.append(image)
            if len(images) == 512:
                predictions = model.predict(np.array(images))
                grapheme_root_predictions.extend(predictions[:, :168].argmax(axis=1))
                vowel_diacritic_predictions.extend(predictions[:, 168:179].argmax(axis=1))
                consonant_diacritic_predictions.extend(predictions[:, 179:].argmax(axis=1))
                images = []
        predictions = model.predict(np.array(images))
        grapheme_root_predictions.extend(predictions[:, :168].argmax(axis=1))
        vowel_diacritic_predictions.extend(predictions[:, 168:179].argmax(axis=1))
        consonant_diacritic_predictions.extend(predictions[:, 179:].argmax(axis=1))
        return pd.DataFrame([
            self.images['image_id'].values, grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions
        ], index=['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).T.set_index('image_id')

    def recall(self, model):
        predictions = self.make_predictions(model).sort_index()
        validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()
        scores = []
        for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
            y_true_subset = validIds[component].values.astype(int)
            y_pred_subset = predictions[component].values.astype(int)
            scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
        return round(np.average(scores, weights=[2,1,1]), 5)


class MultiOutputImageGenerator(Sequence):

    def __init__(self, images, batch_size, is_train):
        self.images = images
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(len(self.images) / self.batch_size)

    def __getitem__(self, idx):

        # if self.is_train:
        #     n_grapheme = int(0.9 * self.batch_size)
        #     n_vowel = int((self.batch_size - n_grapheme) / 2) + 1
        #     n_consonant = self.batch_size - (n_grapheme + n_vowel)
        #     batch_images = pd.concat([
        #         self.images.sample(n=n_grapheme, weights='grapheme_root_weight'),
        #         self.images.sample(n=n_vowel, weights='vowel_diacritic_weight'),
        #         self.images.sample(n=n_consonant, weights='consonant_diacritic_weight')
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
            if self.is_train:
                x = augmentor(image=x)['image']
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
            image = crop_and_resize_image(get_image(image_id))
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
    train_df['grapheme_root_weight'] = 1
    train_df['vowel_diacritic_weight'] = 1
    train_df['consonant_diacritic_weight'] = 1

    train_generator = MultiOutputImageGenerator(train_df, batch_size, True)
    valid_generator = MultiOutputImageGenerator(valid_df, batch_size, False)
    return train_generator, valid_generator
