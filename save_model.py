import cv2
import joblib
import numpy as np
import pandas as pd
from generators import get_data_generators
from algorithms import get_model
from sklearn.metrics import recall_score, multilabel_confusion_matrix

IMAGES = joblib.load('data/original_images')

df = pd.read_csv('data/train.csv')
splits = pd.read_csv('splits/split1/split.csv')
valid_ids = list(splits[splits['split'] == 'valid']['image_id'])
valid_df = df[df['image_id'].isin(valid_ids)].reset_index(drop=True)
model = connect_simple_head(*get_b0_backbone())
model.load_weights('results/split1/train_full.h5')
model.save("results/split1/model.h5")


def scale_values(image):
    image = image / np.percentile(image, 99)
    return image.clip(0, 1)


grapheme_root_predictions = {}
vowel_diacritic_predictions = {}
consonant_diacritic_predictions = {}
for image_id in valid_ids:
    x = IMAGES[image_id].copy()
    x = cv2.resize(x, (128, 128))
    x = x / np.percentile(x, 99)
    x = x.clip(0, 1)
    image = np.stack([x, x, x], axis=2)
    output = model.predict(np.expand_dims(image, 0))
    grapheme_root_predictions[image_id] = output[0]
    vowel_diacritic_predictions[image_id] = output[1]
    consonant_diacritic_predictions[image_id] = output[2]


def get_predictions(weights, y_true, y_pred):
    predictions = []
    for image_id, y in y_true.iteritems():
        pred = y_pred[image_id][0].copy()
        pred = np.argmax(pred * weights)
        predictions.append(pred)
    return predictions

def calculate_recall(weights, y_true, y_pred):
    predictions = []
    for image_id, y in y_true.iteritems():
        pred = y_pred[image_id][0].copy()
        pred = np.argmax(pred * weights)
        predictions.append(pred)

    return round(recall_score(y_true.values, predictions, average='macro'), 5)


def random_optimize(initial_weights, y_true, y_pred, step_size, improvement_limit):

    since_improvement = 0
    n_classes = y_true.max()
    best_weights = [w for w in initial_weights]
    best_value = calculate_recall(best_weights, y_true, y_pred)
    while since_improvement <= improvement_limit:
        new_weights = [w for w in best_weights]
        index = np.random.randint(n_classes)
        if np.random.rand() < 0.5:
            new_weights[index] += step_size
        else:
            new_weights[index] -= step_size
        new_weights = [round(w, 2) for w in new_weights]
        score = calculate_recall(new_weights, y_true, y_pred)
        if score > best_value:
            best_weights = [w for w in new_weights]
            best_value = score
        else:
            since_improvement += 1



    return best_weights, best_value



Y_true = valid_df.set_index('image_id')['vowel_diacritic']
Y_pred = vowel_diacritic_predictions
vweights = [1 for i in range(Y_true.max() + 1)]
print('Starting value:', calculate_recall(vweights, Y_true, Y_pred))
vweights, new_value = random_optimize(vweights, Y_true, Y_pred, 0.4, 100)
print(new_value)
vweights, new_value = random_optimize(vweights, Y_true, Y_pred, 0.2, 100)
print(new_value)
vweights, new_value = random_optimize(vweights, Y_true, Y_pred, 0.1, 100)
print(new_value)
vweights, new_value = random_optimize(vweights, Y_true, Y_pred, 0.02, 100)
print(new_value)
vweights

Y_true = valid_df.set_index('image_id')['consonant_diacritic']
Y_pred = consonant_diacritic_predictions
cweights = [1 for i in range(Y_true.max() + 1)]
print('Starting value:', calculate_recall(cweights, Y_true, Y_pred))
cweights, new_value = random_optimize(cweights, Y_true, Y_pred, 0.4, 100)
print(new_value)
cweights, new_value = random_optimize(cweights, Y_true, Y_pred, 0.2, 100)
print(new_value)
cweights, new_value = random_optimize(cweights, Y_true, Y_pred, 0.1, 100)
print(new_value)
cweights, new_value = random_optimize(cweights, Y_true, Y_pred, 0.02, 100)
print(new_value)

Y_true = valid_df.set_index('image_id')['grapheme_root']
Y_pred = grapheme_root_predictions
gweights = [1 for i in range(Y_true.max() + 1)]
print('Starting value:', calculate_recall(gweights, Y_true, Y_pred))
gweights, new_value = random_optimize(gweights, Y_true, Y_pred, 0.4, 200)
print(new_value)
gweights, new_value = random_optimize(gweights, Y_true, Y_pred, 0.2, 200)
print(new_value)
gweights, new_value = random_optimize(gweights, Y_true, Y_pred, 0.1, 200)
print(new_value)
gweights, new_value = random_optimize(gweights, Y_true, Y_pred, 0.02, 200)
print(new_value)


def weighted_recall(grapheme_weights, vowel_weights, consonant_weights):

    scores = []
    grapheme_pred = get_predictions(grapheme_weights, valid_df.set_index('image_id')['grapheme_root'], grapheme_root_predictions)
    vowel_pred = get_predictions(vowel_weights, valid_df.set_index('image_id')['vowel_diacritic'], vowel_diacritic_predictions)
    consonant_pred = get_predictions(consonant_weights, valid_df.set_index('image_id')['consonant_diacritic'], consonant_diacritic_predictions)
    y_pred = pd.DataFrame([grapheme_pred, vowel_pred, consonant_pred], columns=valid_df.index, index=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).T

    for component in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        y_true_subset = valid_df[component].values.astype(int)
        y_pred_subset = y_pred[component].values.astype(int)
        scores.append(recall_score(y_true_subset, y_pred_subset, average='macro'))

    return round(np.average(scores, weights=[2,1,1]), 5), round(scores[0], 5), round(scores[1], 5), round(scores[2], 5)


weighted_recall([1 for i in range(168)], [1 for i in range(11)], [1 for i in range(7)])
weighted_recall(gweights, vweights, cweights)
