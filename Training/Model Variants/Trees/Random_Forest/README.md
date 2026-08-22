# Random Forest Individual Hyperparameter Trials
Run order:
1. 00_RF_Baseline.ipynb
2. 01_RF_NEstimators_Trial.ipynb — changes only `n_estimators`
3. 02_RF_MaxDepth_Trial.ipynb — changes only `max_depth`
4. 03_RF_MinSamplesSplit_Trial.ipynb — changes only `min_samples_split`
5. 04_RF_MinSamplesLeaf_Trial.ipynb — changes only `min_samples_leaf`
6. 05_RF_MaxFeatures_Trial.ipynb — changes only `max_features`
7. 06_RF_Trial_Comparison.ipynb
8. 07_RF_Final_Selected_Model.ipynb — enter chosen settings and evaluate test once

Suggested Git strategy: commit the baseline, then commit each completed individual trial separately after its real outputs are produced, then commit the comparison and final selected model.
Do not choose hyperparameters from test-set performance.
