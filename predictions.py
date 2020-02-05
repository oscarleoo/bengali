import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

from algorithms import get_b0_backbone, get_b1_backbone, connect_simple_head
from generators import get_data_generators



train_generator, valid_generator = get_data_generators('split1', 64)

backbone, backbone_output = get_b1_backbone()
model = connect_simple_head(backbone, backbone_output)

model.save('results/b1_simple/model.h5')
with open("results/b1_simpe/model.json", "w") as json_file:
    json_file.write(model_json)

model.load_weights('results/b1_simple/final_step_weights.h5')
predictions = valid_generator.make_predictions(model)
trainIds = pd.read_csv('data/train.csv').set_index('image_id')
trainIds.head()

predictions = predictions.sort_index()
validIds = trainIds[trainIds.index.isin(predictions.index)].sort_index()
validIds.head(5)


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
