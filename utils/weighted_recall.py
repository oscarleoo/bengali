import numpy as np
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import recall_score, multilabel_confusion_matrix

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)


def calculate_class_weights(y_true, y_pred, title):
    MCM = multilabel_confusion_matrix(
        y_true[title].values.astype(int),
        y_pred[title].values.astype(int)
    )
    true_positives = MCM[:, 1, 1]
    true_sum = true_positives + MCM[:, 1, 0]
    class_recall = (true_positives / true_sum)
    class_recall = [(i, r) for i, r in zip([i for i in range(len(class_recall))], class_recall)]
    print('\n ==>', title)
    for i, r in class_recall:
        print('class {}: {}'.format(i, round(r, 4)))


def calculate_recall(y_true, y_pred):
    scores = []
    for component in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        y_true_subset = y_true[component].values.astype(int)
        y_pred_subset = y_pred[component].values.astype(int)
        scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
    return round(np.average(scores, weights=[2,1,1]), 5), round(scores[0], 5), round(scores[1], 5), round(scores[2], 5)


class WeightedRecall(Callback):

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def on_epoch_end(self, batch, logs={}):
        predictions = self.valid.make_predictions(self.model).sort_index()
        validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()
        calculate_class_weights(validIds, predictions, 'grapheme_root')
        calculate_class_weights(validIds, predictions, 'vowel_diacritic')
        calculate_class_weights(validIds, predictions, 'consonant_diacritic')
        score, gr_score, vd_score, cd_score = calculate_recall(validIds, predictions)
        print('\n ==> Weighted Recal Score: {} ({} - {} - {})'.format(score, gr_score, vd_score, cd_score))
        return
