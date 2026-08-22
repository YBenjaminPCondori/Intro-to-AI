# CNN: Random Search + Genetic Algorithm

These notebooks are derived directly from the uploaded source notebook: `Convolutional Neural Network Training Script.ipynb`.

## Run order
1. `00_CNN_Baseline.ipynb` — fixed reference configuration; validation only.
2. `01_CNN_Random_Search.ipynb` — 12 reproducible random configurations; validation F1 selection.
3. `02_CNN_Genetic_Algorithm.ipynb` — population 8, 4 generations, tournament selection, uniform crossover, 20% per-gene mutation, elitism; validation F1 fitness.
4. `03_CNN_Optimisation_Comparison.ipynb` — compares the two optimisers and writes `selected_hyperparameters.json`.
5. `04_CNN_Final_Selected_Model.ipynb` — loads the selected hyperparameters and evaluates the held-out test set once.

## Leakage control
Random Search and the Genetic Algorithm use the training and validation partitions only. The final test set remains untouched until notebook 04.

## Search space
```python
{'filters': [[16, 32], [32, 64], [64, 128], [32, 64, 128]], 'kernel_size': [[3, 3], [5, 3]], 'dense_units': [32, 64, 128], 'dropout': [0.2, 0.3, 0.4, 0.5], 'learning_rate': [0.0001, 0.0003, 0.001, 0.003], 'batch_size': [32, 64, 128]}
```

The generated notebooks contain no fabricated results. They must be run against the dataset paths already defined by the uploaded source notebook.
