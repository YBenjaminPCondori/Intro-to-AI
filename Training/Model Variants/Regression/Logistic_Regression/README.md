# Logistic Regression Individual Hyperparameter Trials
Run order:
1. 00_LR_Baseline.ipynb
2. 01_LR_C_Trial.ipynb — changes only `C`
3. 02_LR_Penalty_Trial.ipynb — changes only `penalty`
4. 03_LR_Solver_Trial.ipynb — changes only `solver`
5. 04_LR_ClassWeight_Trial.ipynb — changes only `class_weight`
6. 05_LR_Trial_Comparison.ipynb
7. 06_LR_Final_Selected_Model.ipynb — enter chosen settings and evaluate test once

Suggested Git strategy: commit the baseline, then commit each completed individual trial separately after its real outputs are produced, then commit the comparison and final selected model.
Do not choose hyperparameters from test-set performance.
