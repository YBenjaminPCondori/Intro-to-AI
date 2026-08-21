# XGBoost Individual Hyperparameter Trials
Run order:
1. 00_XGB_Baseline.ipynb
2. 01_XGB_NEstimators_Trial.ipynb — changes only `n_estimators`
3. 02_XGB_MaxDepth_Trial.ipynb — changes only `max_depth`
4. 03_XGB_LearningRate_Trial.ipynb — changes only `learning_rate`
5. 04_XGB_Subsample_Trial.ipynb — changes only `subsample`
6. 05_XGB_ColsampleBytree_Trial.ipynb — changes only `colsample_bytree`
7. 06_XGB_Trial_Comparison.ipynb
8. 07_XGB_Final_Selected_Model.ipynb — enter chosen settings and evaluate test once

Suggested Git strategy: commit the baseline, then commit each completed individual trial separately after its real outputs are produced, then commit the comparison and final selected model.
Do not choose hyperparameters from test-set performance.
