# MLAAD MFCC model variants

This folder implements binary audio deepfake detection with:

- `0 = bona-fide`
- `1 = synthetic/deepfake`

The selected core models are:

- SVM, using fixed-length aggregated MFCC vectors
- MLP, using fixed-length aggregated MFCC vectors
- CNN, using MFCC coefficient-by-time maps with a channel dimension

## Notebooks

1. `Download Dataset Script.ipynb` optionally downloads the MLAAD synthetic subset.
2. `MLAAD Scan & MFCC Analysis Cache.ipynb` is now an optional pre-flight notebook. It scans data, checks class balance and file quality, previews MFCC shapes, and saves simple manifest tables.
3. `Support Vector Machine Training Script.ipynb` trains and evaluates the SVM baseline.
4. `Multi Layer Perceptron Training Sciprt.ipynb` trains and evaluates the MLP baseline.
5. `Convolutional Neural Network Training Script.ipynb` trains and evaluates the CNN model.
6. `Optional Random Forest Baseline.ipynb` trains an optional tree-based baseline on aggregated MFCC features.
7. `Optional Logistic Regression Baseline.ipynb` trains an optional simple linear classifier on aggregated MFCC features.
8. `Optional XGBoost Baseline.ipynb` trains an optional gradient-boosted tree classifier on aggregated MFCC features.
9. `Optional LSTM Baseline.ipynb` trains an optional sequence-model baseline on MFCC time sequences.
10. `Optional PCA MFCC Feature Analysis.ipynb` uses PCA for optional feature inspection only.

The three model notebooks are self-contained. Each one defines its own dataset loading, preprocessing, MFCC extraction, splitting, evaluation, plotting, and output-saving code in visible notebook cells. They do not rely on a shared MFCC cache or a custom helper module.

The optional notebooks are also self-contained. They are included for comparison and feature analysis, but they are not part of the selected core model set unless you decide to keep them.

## Evaluation policy

- Scaling is fitted on training data only.
- Hyperparameter and architecture selection use validation data only.
- The held-out test split is loaded only after validation-based selection.
- Core metrics are accuracy, precision, recall, F1-score, and confusion matrix.
- MLP and CNN additionally record training loss, validation loss, training accuracy, validation accuracy, best epoch, epochs trained, and early-stopping status.

## Optional techniques

- Random Forest is available as an optional additional baseline.
- Logistic Regression is available as an optional simple linear classifier. It is used instead of Linear Regression because the task is binary classification.
- XGBoost is available as an optional boosted-tree baseline.
- LSTM is available only as an optional comparison notebook. It is not one of the selected core models.
- PCA is available as optional preprocessing / feature analysis only. It is not a classifier and is not required for SVM, MLP, or CNN.

## Final run notes

For final coursework runs, configure `SYNTHETIC_AUDIO_DIR` and `BONA_FIDE_AUDIO_DIR` in each notebook. `MAX_FILES_PER_CLASS` can be set to a small number for a quick check, but leave it as `None` for final experiments.
