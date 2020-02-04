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


counts = trainIds.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).count()['image_id']

counts.sort_values()
