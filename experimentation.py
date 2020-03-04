import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import cv2


train = pd.read_csv('data/train.csv')
train.head()

train.shape

###################################
#       SETTINGS
###################################


trainIds = pd.read_parquet('data/train_image_data_0.parquet')




trainIds['grapheme_root'].value_counts().shape
trainIds['vowel_diacritic'].value_counts().shape
trainIds['consonant_diacritic'].value_counts().shape

round(np.sqrt(168) / (np.sqrt(168) + np.sqrt(11) + np.sqrt(7)), 3)
round(np.sqrt(11) / (np.sqrt(168) + np.sqrt(11) + np.sqrt(7)), 3)
round(np.sqrt(7) / (np.sqrt(168) + np.sqrt(11) + np.sqrt(7)), 3)

counts = trainIds.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).count()['image_id']

counts.sort_values()


trainIds['grapheme_root'].unique().shape
trainIds['vowel_diacritic'].unique().shape
trainIds['consonant_diacritic'].unique().shape


168 * 11 * 7

trainIds[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].drop_duplicates().shape

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


trainIds


trainIds.sample(n=1000, weights='vowel_diacritic_weight', replace=True)

train_ids
np.sqrt(168)


import pickle

with open('results/b1_simple/final_step', 'rb') as f:
    hmm = pickle.load(f)

hmm

max(hmm['val_grapheme_root_categorical_accuracy'])
max(hmm['val_vowel_diacritic_categorical_accuracy'])
max(hmm['val_consonant_diacritic_categorical_accuracy'])




train = pd.concat([
    pd.read_parquet('data/train_image_data_0.parquet'),
    pd.read_parquet('data/train_image_data_1.parquet'),
    pd.read_parquet('data/train_image_data_2.parquet'),
    pd.read_parquet('data/train_image_data_3.parquet')
]).set_index('image_id', drop=True)


train.head()
import cv2
import joblib
len(original_images)
original_images = {}
for i, row in train.iterrows():
    if i in original_images.keys():
        continue
    image = 255 - row.values
    image = image.reshape(137, 236)
    image = image.astype(np.uint8)
    original_images[i] = image

joblib.dump(original_images, 'data/original_images')

print('hej')



plt.imshow(image)





plt.imshow(image.clip(0, 10))
