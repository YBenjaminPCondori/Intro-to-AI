# MLP staged TensorFlow/Keras workflow

Run the notebooks in this order:

1. `00_MLP_Data_Preparation.ipynb`
2. `01_MLP_Baseline.ipynb`
3. `02_MLP_Activation_Trial.ipynb`
4. `03_MLP_Optimizer_Trial.ipynb`
5. `04_MLP_Dropout_Regularisation_Trial.ipynb`
6. `05_MLP_Architecture_Trial.ipynb`
7. `06_MLP_LearningRate_BatchSize_Trial.ipynb`
8. `07_MLP_Controlled_Trial_Comparison.ipynb`
9. `08_MLP_Random_Search.ipynb`
10. `09_MLP_Genetic_Algorithm.ipynb`
11. `10_MLP_HPO_Comparison.ipynb`
12. `11_MLP_Final_Selected_Model.ipynb`

Notebook `00` creates the final MLP split and one MLP MFCC cache. Notebooks `01` to `10` use train and validation data only. Notebook `11` is the only MLP notebook that evaluates the held-out test set.

Old broad-sequence notebooks were archived under `outputs/mlp/pilot_legacy/`.
