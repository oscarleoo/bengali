import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

###################################
#       SETTINGS
###################################


trainIds = pd.read_csv('data/train.csv')
trainIds.head()


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
