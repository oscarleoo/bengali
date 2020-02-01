import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

###################################
#       SETTINGS
###################################

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < 236 - 13) else 236
    ymax = ymax + 10 if (ymax < 137 - 10) else 137
    img = img0[ymin:ymax,xmin:xmax]
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

import os
import joblib

def preprocess_images(types):

    if os.path.exists('data/images'):
        IMAGES = joblib.load('data/images')
    else:
        IMAGES = {}

    FILES = []
    for file_type in types:
        FILES.extend([
            '{}_image_data_0.parquet'.format(file_type),
            '{}_image_data_1.parquet'.format(file_type),
            '{}_image_data_2.parquet'.format(file_type),
            '{}_image_data_3.parquet'.format(file_type),
        ])

    for file_path in FILES:
        df = pd.read_parquet('data/{}'.format(file_path))
        df = df[~df['image_id'].isin(IMAGES.keys())]
        print(file_path, len(df))
        for idx in range(len(df)):
            image_id = df.iloc[idx, 0]
            if image_id in IMAGES.keys():
                continue
            img = 255 - df.iloc[idx, 1:].values.reshape(137, 236).astype(np.uint8)
            img = (img * (255.0 / img.max())).astype(np.uint8)
            img = crop_resize(img)
            IMAGES[image_id] = img
        del df
    joblib.dump(IMAGES, 'data/images')
    return IMAGES

images = preprocess_images(['train', 'test'])

IMAGES = joblib.load('data/images')

plt.imshow(IMAGES['Train_9'], cmap='gray')

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

def show_examples(grapheme_root, vowel_diacritic, consonant_diacritic, n=10):

    ids = trainIds[trainIds['grapheme_root'] == grapheme_root]
    ids = ids[ids['vowel_diacritic'] == vowel_diacritic]
    ids = ids[ids['consonant_diacritic'] == consonant_diacritic]

    for i in range(n):
        random_id = np.random.choice(ids.index)
        example = IMAGES[random_id].copy()
        binary1 = example >= 80
        binary2 = example >= 150
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 3))

        axes[0].imshow(example, cmap='gray')
        axes[1].imshow(binary1, cmap='gray')
        axes[2].imshow(binary2, cmap='gray')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        plt.show()


show_examples(159, 0, 0)

import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence

def get_image(image_id):

    x = 255 - train.loc[image_id].values.reshape(137, 236).astype(np.uint8)
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
        return 100

    def __getitem__(self, idx):

        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]

        X = np.zeros((self.batch_size, 137, 236, 3))
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

ids = list(trainIds.index)
np.random.shuffle(ids)
len(ids)
train_ids = ids[:200000]
valid_ids = ids[200000:]
train_generator = ImageGenerator(train_ids, 64, True)
valid_generator = ImageGenerator(valid_ids, 128, True)

backbone = efn.EfficientNetB0(input_shape=(137, 236, 3), include_top=False,  weights='imagenet')
x = GlobalAveragePooling2D()(backbone.output)

grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(x)
vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(x)
consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(x)
model = Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])

losses = {
	'grapheme_root': 'categorical_crossentropy',
	'vowel_diacritic': 'categorical_crossentropy',
    'consonant_diacritic': 'categorical_crossentropy'
}

model.compile(optimizer='adam', loss=losses, metrics=["accuracy"])

history = model.fit_generator(
    train_generator, steps_per_epoch=100, epochs=5,
    validation_data=valid_generator, validation_steps=int(len(valid_ids) / 32),
)


def make_predictions(image_ids):
    grapheme_root_predictions = []
    vowel_diacritic_predictions = []
    consonant_diacritic_predictions = []
    images = []
    for image_id in image_ids:
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
        valid_ids[:100], grapheme_root_predictions, vowel_diacritic_predictions, consonant_diacritic_predictions
    ], index=['image_id', 'grapheme_root', 'consonant_diacritic', 'vowel_diacritic']).T.set_index('image_id')

def get_true_values(image_ids):

    for img_id in

grp, vdp, cdp = make_predictions(valid_ids[:100])

submission = pd.DataFrame([valid_ids[:100], grp, vdp, cdp], index=['image_id', 'grapheme_root', 'consonant_diacritic', 'vowel_diacritic']).T.set_index('image_id')

submission
trainIds

trueValues = trainIds.loc[submission.index]
trueValues


import numpy as np
import sklearn.metrics
trueValues[trueValues[component] == component]
trueValues

scores = []
for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
    y_true_subset = trueValues[trueValues[component] == component]['target'].values
    y_pred_subset = submission[submission[component] == component]['target'].values
    scores.append(sklearn.metrics.recall_score(y_true_subset, y_pred_subset, average='macro'))
final_score = np.average(scores, weights=[2,1,1])
