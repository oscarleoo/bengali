import numpy as np
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import recall_score, multilabel_confusion_matrix

trainIds = pd.read_csv('data/train.csv')
trainIds = trainIds.set_index('image_id', drop=True)


def calculate_class_weights(y_true, y_pred, title, alpha):
    MCM = multilabel_confusion_matrix(
        y_true[title].values.astype(int),
        y_pred[title].values.astype(int)
    )
    true_positives = MCM[:, 1, 1]
    true_sum = true_positives + MCM[:, 1, 0]
    class_recall = (true_positives / true_sum) + alpha
    class_recall = 1 / class_recall
    class_recall = class_recall / (true_sum ** (1/4))
    class_recall = class_recall / class_recall.sum()
    class_index = [i for i in range(len(class_recall))]
    return pd.DataFrame([class_index, class_recall], index=[title, '{}_weight'.format(title)]).T


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
        gr_weights = calculate_class_weights(validIds, predictions, 'grapheme_root', 1/168)
        vc_weights = calculate_class_weights(validIds, predictions, 'vowel_diacritic', 1/11)
        cd_weights = calculate_class_weights(validIds, predictions, 'consonant_diacritic', 1/7)
        self.train.update_weights(gr_weights, vc_weights, cd_weights)
        score, gr_score, vd_score, cd_score = calculate_recall(validIds, predictions)
        print('==> Weighted Recal Score: {} ({} - {} - {})'.format(score, gr_score, vd_score, cd_score))
        return
