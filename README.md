# Bengali Challenge

Bengali is the language in Bangladesh. It has 11 vowels, 38 consonants and 18 diacritics/accents.

Bengali characters are called graphemes and is a combination of a grapheme_root, a vowel_diacritic, and a consonant_diacritic

## Todo
- Investigate the distribution of grapheme_root, vowel_diacritic, and consonant_diacritic
- Investigate the looks of grapheme_root, vowel_diacritic, and consonant_diacritic

## Strategy

I want to use a simple and clean approach to solving this challenge.

#### Sampling

Since the training dataset is large, I can't sample based on loss. I plan to sample in two steps:
1. Sample X images based on class distribution
2. Calculate loss for the X images
3. Sample each batch based on loss

## Experiments

### Additional Cropping
- loss: 0.1620
- grapheme_root_loss: 0.1073
- vowel_diacritic_loss: 0.0274
- consonant_diacritic_loss: 0.0273
- grapheme_root_categorical_accuracy: 0.9669
- vowel_diacritic_categorical_accuracy: 0.9925
- consonant_diacritic_categorical_accuracy: 0.9913
- val_loss: 0.4134
- val_grapheme_root_loss: 0.2685
- val_vowel_diacritic_loss: 0.0759
- val_consonant_diacritic_loss: 0.0795
- val_grapheme_root_categorical_accuracy: 0.9266
- val_vowel_diacritic_categorical_accuracy: 0.9813
- val_consonant_diacritic_categorical_accuracy: 0.9795
