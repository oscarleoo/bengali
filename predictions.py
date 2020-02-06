import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

from algorithms import get_b0_backbone, get_b1_backbone, connect_simple_head
from generators import get_data_generators
from keras.models import load_model


train_generator, valid_generator = get_data_generators('split1', 64)
valid_generator.images = valid_generator.images.sample(n=500)
model = load_model('model.h5')
predictions = valid_generator.make_predictions(model)
predictions = predictions.sort_index()

trainIds = pd.read_csv('data/train.csv').set_index('image_id')
validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()


validIds.head(5)

predictions.head()


from sklearn.metrics import  multilabel_confusion_matrix

def calculate_class_weights(y_true, y_pred, title, alpha):
    MCM = multilabel_confusion_matrix(
        y_true[title].values.astype(int),
        y_pred[title].values.astype(int)
    )
    true_positives = MCM[:, 1, 1]
    true_sum = true_positives + MCM[:, 1, 0]
    class_recall = (true_positives / true_sum) + alpha
    class_recall = 1 / class_recall
    class_recall = class_recall / class_recall.sum()
    class_index = [i for i in range(len(class_recall))]
    return pd.DataFrame([class_index, class_recall], index=[title, '{}_weight'.format(title)]).T

calculate_class_weights(validIds, predictions, 'vowel_diacritic', 0.1)

true_sum

1 / class_recall

tp_sum
true_sum
pred_sum

validIds['vowel_diacritic'].value_counts().sort_index()
tp_sum / true_sum


validIds['consonant_diacritic'].values.astype(int)
for i in range()


hmm.shape

    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]


classification_report(validIds['vowel_diacritic'].values.astype(int), predictions['vowel_diacritic'].values.astype(int))


scores = []
for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
    y_true_subset = validIds[component].values.astype(int)
    y_pred_subset = predictions[component].values.astype(int)
    scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))
round(np.average(scores, weights=[2,1,1]), 4)


#   b0_simple_pretrain => 0.9569
#   b0_simple_final_step => 0.9595
#   b1_simple_pretrain => 0.9581
#   b1_simple_pretrain => 0.9594
#
