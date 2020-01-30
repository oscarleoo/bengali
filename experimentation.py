import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.concat([
    pd.read_parquet('bengali/data/train_image_data_0.parquet'),
    pd.read_parquet('bengali/data/train_image_data_1.parquet'),
    pd.read_parquet('bengali/data/train_image_data_2.parquet'),
    pd.read_parquet('bengali/data/train_image_data_3.parquet')
])
train = train.set_index('image_id', drop=True)
trainIds = pd.read_csv('bengali/data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)

def show_examples(grapheme_root, vowel_diacritic, consonant_diacritic, n=10):

    ids = trainIds[trainIds['grapheme_root'] == grapheme_root]
    ids = ids[ids['vowel_diacritic'] == vowel_diacritic]
    ids = ids[ids['consonant_diacritic'] == consonant_diacritic]

    for i in range(n):
        random_id = np.random.choice(ids.index)
        example = 255 - train.loc[random_id].values.reshape(137, 236).astype(np.uint8)
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

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

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
            x = 255 - train.loc[image_id].values.reshape(137, 236).astype(np.uint8)
            x1 = x >= 80
            x2 = x >= 150
            x = x - x.min()
            x = x / x.max()

            X[i] = np.stack([x, x1, x2], axis=2)
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
train_ids = ids[:150000]
valid_ids = ids[150000:]
150000/512
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
    predictions, images = [], []
    for image_id in image_ids:
        images.append(get_image(image_id))
        if len(images) == 512:
            predictions.extend(model.predict(np.array(images)))
            images = []
    predictions.extend(model.predict(np.array(images)))
    return predictions


predictions = make_predictions(valid_ids[:100])
predictions[0].shape

predictions[0].argmax(axis=1).shape

len(predictions)
predictions[0].shape

predictions = model.predict_generator(valid_generator)
predictions[0].shape



import numpy as np
import sklearn.metrics

scores = []
for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
    y_true_subset = solution[solution[component] == component]['target'].values
    y_pred_subset = submission[submission[component] == component]['target'].values
    scores.append(sklearn.metrics.recall_score(y_true_subset, y_pred_subset, average='macro'))
final_score = np.average(scores, weights=[2,1,1])
