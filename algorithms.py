import tensorflow as tf
import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Lambda
from keras.models import Model


gm_exp = tf.Variable(3.0, dtype = tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)),
                        axis = [1, 2],
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool


def get_model():

    backbone = efn.EfficientNetB0(input_shape=(96, 96, 3), include_top=False,  pooling=None, classes=None, weights='imagenet')

    for layer in backbone.layers:
        layer.trainable = True

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    gem = lambda_layer(backbone.output)

    intermediate = Dense(512, activation='relu', name='vowel_diacritic')(gem)
    output = Dense(186, activation='sigmoid', name='output')(intermediate)

    return Model(backbone.input, output)
