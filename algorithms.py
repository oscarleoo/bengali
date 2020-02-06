import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


def get_b0_backbone():
    backbone = efn.EfficientNetB0(input_shape=(64, 64, 3), include_top=False,  weights='imagenet')
    global_average = GlobalAveragePooling2D()(backbone.output)
    # backbone_output = Dropout(0.5)(global_average)
    return backbone, global_average


def get_b1_backbone():
    backbone = efn.EfficientNetB1(input_shape=(128, 128, 3), include_top=False,  weights='imagenet')
    global_average = GlobalAveragePooling2D()(backbone.output)
    backbone_output = Dropout(0.5)(global_average)
    return backbone, backbone_output


def get_b2_backbone():
    backbone = efn.EfficientNetB2(input_shape=(128, 128, 3), include_top=False,  weights='imagenet')
    global_average = GlobalAveragePooling2D()(backbone.output)
    backbone_output = Dropout(0.5)(global_average)
    return backbone, backbone_output


def get_b3_backbone():
    backbone = efn.EfficientNetB3(input_shape=(128, 128, 3), include_top=False,  weights='imagenet')
    global_average = GlobalAveragePooling2D()(backbone.output)
    backbone_output = Dropout(0.5)(global_average)
    return backbone, backbone_output

def connect_simple_head(backbone, backbone_output):

    # grapheme_root_head
    grapheme_root_dense = Dense(512, activation='relu', name='grapheme_root_dense')(backbone_output)
    grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(grapheme_root_dense)

    # vowel_diacritic_head
    vowel_diacritic_dense = Dense(32, activation='relu', name='vowel_diacritic_dense')(backbone_output)
    vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(vowel_diacritic_dense)

    # consonant_diacritic_head
    consonant_diacritic_dense = Dense(32, activation='relu', name='consonant_diacritic_dense')(backbone_output)
    consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(consonant_diacritic_dense)

    return Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])


def connect_medium_head(backbone, backbone_output):

    # grapheme_root_head
    grapheme_root_dense = Dense(128, activation='relu', name='grapheme_root_dense')(backbone_output)
    grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(grapheme_root_dense)

    # vowel_diacritic_head
    vowel_diacritic_dense = Dense(64, activation='relu', name='vowel_diacritic_dense')(backbone_output)
    vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(vowel_diacritic_dense)

    # consonant_diacritic_head
    consonant_diacritic_dense = Dense(64, activation='relu', name='consonant_diacritic_dense')(backbone_output)
    consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(consonant_diacritic_dense)

    return Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])


def connect_complex_head(backbone, backbone_output):

    # grapheme_root_head
    grapheme_root_dense1 = Dense(256, activation='relu', name='grapheme_root_dense1')(backbone_output)
    grapheme_root_dense2 = Dense(128, activation='relu', name='grapheme_root_dense2')(grapheme_root_dense1)
    grapheme_root_head = Dense(168, activation='softmax', name='grapheme_root')(grapheme_root_dense2)

    # vowel_diacritic_head
    vowel_diacritic_dense1 = Dense(32, activation='relu', name='vowel_diacritic_dense1')(backbone_output)
    vowel_diacritic_dense2 = Dense(32, activation='relu', name='vowel_diacritic_dense2')(vowel_diacritic_dense1)
    vowel_diacritic_head = Dense(11, activation='softmax', name='vowel_diacritic')(vowel_diacritic_dense2)

    # consonant_diacritic_head
    consonant_diacritic_dense1 = Dense(32, activation='relu', name='consonant_diacritic_dense1')(backbone_output)
    consonant_diacritic_dense2 = Dense(32, activation='relu', name='consonant_diacritic_dense2')(consonant_diacritic_dense1)
    consonant_diacritic_head = Dense(7, activation='softmax', name='consonant_diacritic')(consonant_diacritic_dense2)

    return Model(backbone.input, outputs=[grapheme_root_head, vowel_diacritic_head, consonant_diacritic_head])
