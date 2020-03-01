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
    class_recall.sort(key=lambda x: x[0])
    return class_recall


def calculate_recall(y_true, y_pred):
    scores = []
    for component in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        y_true_subset = y_true[component].values.astype(int)
        y_pred_subset = y_pred[component].values.astype(int)
        scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
    return round(np.average(scores, weights=[2,1,1]), 5), round(scores[0], 5), round(scores[1], 5), round(scores[2], 5)


def print_recall(trainRecall, validRecall, title):

    print('\n{} Recall'.format(title))
    trainRecall = {i: tr for i, tr in trainRecall}
    validRecall.sort(key=lambda x: x[1])
    for i, vr in validRecall:
        print('{}: {} ({})'.format(i, round(vr, 4), round(trainRecall[i], 4)))



class WeightedRecall(Callback):

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def on_epoch_end(self, batch, logs={}):
        valid_predictions = self.valid.make_predictions(self.model).sort_index()
        train_predictions = self.train.make_predictions(self.model).sort_index()

        valid_ids = trainIds[trainIds.index.isin(valid_predictions.index)].sort_index()
        train_ids = trainIds[trainIds.index.isin(train_predictions.index)].sort_index()

        validGrapheme = calculate_class_weights(valid_ids, valid_predictions, 'grapheme_root')
        trainGrapheme = calculate_class_weights(train_ids, train_predictions, 'grapheme_root')
        print_recall(trainGrapheme, validGrapheme, 'GraphemeRoot')

        validVowel = calculate_class_weights(valid_ids, valid_predictions, 'vowel_diacritic')
        trainVowel = calculate_class_weights(train_ids, train_predictions, 'vowel_diacritic')
        print_recall(trainVowel, validVowel, 'VowelDiacritic')

        validConsonant = calculate_class_weights(valid_ids, valid_predictions, 'consonant_diacritic')
        trainConsonant = calculate_class_weights(train_ids, train_predictions, 'consonant_diacritic')
        print_recall(trainConsonant, validConsonant, 'ConsonantDiacritic')


        valid_score, valid_gr_score, valid_vd_score, valid_cd_score = calculate_recall(valid_ids, valid_predictions)
        train_score, train_gr_score, train_vd_score, train_cd_score = calculate_recall(train_ids, train_predictions)

        print()
        print('==> Weighted Valid Recal Score: {} ({} - {} - {})'.format(valid_score, valid_gr_score, valid_vd_score, valid_cd_score))
        print('==> Weighted Train Recal Score: {} ({} - {} - {})'.format(train_score, train_gr_score, train_vd_score, train_cd_score))
        return
