# LSTM: Random Search + Genetic Algorithm

The LSTM remains optional. Run the shared analysis notebook first, then `00_LSTM_Data_Preparation.ipynb`.

## Run order
1. `00_LSTM_Data_Preparation.ipynb` — loads the shared raw MFCC matrices, transposes them to `T x 40`, fits train-only sequence normalisation, and saves `outputs/lstm_optional/cache/`.
2. `00_LSTM_Baseline.ipynb` — loads the LSTM-ready cache; fixed reference configuration; validation only.
3. `01_LSTM_Random_Search.ipynb` — reproducible random configurations; validation F1 selection.
4. `02_LSTM_Genetic_Algorithm.ipynb` — population search with validation F1 fitness.
5. `03_LSTM_Optimisation_Comparison.ipynb` — compares optimisers and writes selected hyperparameters.
6. `04_LSTM_Final_Selected_Model.ipynb` — loads the selected hyperparameters and evaluates the held-out test set once.

## Leakage control
Random Search and the Genetic Algorithm use the training and validation partitions only. The final test set remains untouched until notebook 04.

## Search space
```python
{'units': [32, 64, 128], 'dropout': [0.2, 0.3, 0.4, 0.5], 'learning_rate': [0.0001, 0.0003, 0.001, 0.003], 'batch_size': [32, 64, 128]}
```

The generated notebooks contain no fabricated results. They load the shared split/cache artifacts and do not rescan audio, create their own split, or call Librosa MFCC extraction.
