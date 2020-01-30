import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.concat([
    pd.read_parquet('data/train_image_data_0.parquet'),
    pd.read_parquet('data/train_image_data_1.parquet'),
    pd.read_parquet('data/train_image_data_2.parquet'),
    pd.read_parquet('data/train_image_data_3.parquet')
]).set_index('image_id', drop=True)
trainIds = pd.read_csv('./data/train.csv')

def show_examples(grapheme_root, vowel_diacritic, consonant_diacritic, n=10):

    ids = trainIds[trainIds['grapheme_root'] == grapheme_root]
    ids = ids[ids['vowel_diacritic'] == vowel_diacritic]
    ids = ids[ids['consonant_diacritic'] == consonant_diacritic]

    for i in range(n):
        random_id = np.random.choice(ids['image_id'])
        example = 255 - train.loc[random_id].values.reshape(137, 236).astype(np.uint8)
        example
        plt.imshow(example, cmap='gray')
        plt.show()


show_examples(159, 0, 0)

trainIds.head()


train.head()


trainIds = trainIds.set_index('image_id', drop=True)

trainIds['consonant_diacritic'].unique().shape

import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence


class ImageGenerator(Sequence):

    def __init__(self, ids, batch_size, is_train):
        self.ids = ids
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):

        X = np.zeros((self.batch_size, 137, 236, 1))
        grapheme_root_Y = np.zeros((self.batch_size, 168))
        vowel_diacritic_Y = np.zeros((self.batch_size, 11))
        consonant_diacritic_Y = np.zeros((self.batch_size, 7))

        for i in range(self.batch_size):
            random_id = np.random.choice(self.ids)
            x = 255 - train.loc[random_id].values.reshape(137, 236).astype(np.uint8)
            x = x - x.min()
            x = x / x.max()

            X[i] = np.expand_dims(x, 2)
            grapheme_root_Y[i][trainIds.loc[random_id]['grapheme_root']] = 1
            vowel_diacritic_Y[i][trainIds.loc[random_id]['vowel_diacritic']] = 1
            consonant_diacritic_Y[i][trainIds.loc[random_id]['consonant_diacritic']] = 1

        return X, {
            'grapheme_root': grapheme_root_Y,
            'vowel_diacritic': vowel_diacritic_Y,
            'consonant_diacritic': consonant_diacritic_Y
        }



backbone = efn.EfficientNetB0(input_shape=(137, 236, 1), include_top=False,  weights=None)
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



train_generator = ImageGenerator(list(trainIds.index), 16, True)

hmm = train_generator.__getitem__(0)

trainIds

trainIds = trainIds[trainIds.index.isin(train.index)]

model.fit_generator(
    train_generator, steps_per_epoch=10, epochs=100
)
