import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, concatenate
from keras.models import Model

def get_b0_backbone():
    backbone = efn.EfficientNetB0(input_shape=(137, 236, 3), include_top=False,  weights='imagenet')
    backbone_output = concatenate([
        GlobalAveragePooling2D()(backbone.output),
        GlobalMaxPooling2D()(backbone.output),
    ])
    return backbone, backbone_output


def get_b1_backbone():
    backbone = efn.EfficientNetB1(input_shape=(64, 64, 3), include_top=False,  weights='imagenet')
    backbone_output = concatenate([
        GlobalAveragePooling2D()(backbone.output),
        GlobalMaxPooling2D()(backbone.output),
    ])
    return backbone, backbone_output


def get_b2_backbone():
    backbone = efn.EfficientNetB2(input_shape=(64, 64, 3), include_top=False,  weights='imagenet')
    backbone_output = GlobalAveragePooling2D()(backbone.output)
    return backbone, backbone_output


def get_b3_backbone():
    backbone = efn.EfficientNetB3(input_shape=(64, 64, 3), include_top=False,  weights='imagenet')
    backbone_output = GlobalAveragePooling2D()(backbone.output)
    return backbone, backbone_output

def connect_simple_head(backbone, backbone_output):

    # grapheme_root_head
    grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(backbone_output)
    vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(backbone_output)
    consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(backbone_output)

    return Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])
