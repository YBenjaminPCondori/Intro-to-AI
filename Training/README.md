# MLAAD MFCC model variants

This project implements binary audio deepfake detection:

- `0 = bona_fide`
- `1 = synthetic/deepfake`

The MLP workflow has been refactored into a staged TensorFlow/Keras experiment sequence. The MLP data-preparation notebook owns corrected metadata parsing, dataset audit, the final frozen split, MFCC extraction, the single MLP feature cache, train-only scaling, and training-only class weights.

## MLP execution order

1. `Model Variants/Neural Networks/MLP/00_MLP_Data_Preparation.ipynb`
2. `Model Variants/Neural Networks/MLP/01_MLP_Baseline.ipynb`
3. `Model Variants/Neural Networks/MLP/02_MLP_Activation_Trial.ipynb`
4. `Model Variants/Neural Networks/MLP/03_MLP_Optimizer_Trial.ipynb`
5. `Model Variants/Neural Networks/MLP/04_MLP_Dropout_Regularisation_Trial.ipynb`
6. `Model Variants/Neural Networks/MLP/05_MLP_Architecture_Trial.ipynb`
7. `Model Variants/Neural Networks/MLP/06_MLP_LearningRate_BatchSize_Trial.ipynb`
8. `Model Variants/Neural Networks/MLP/07_MLP_Controlled_Trial_Comparison.ipynb`
9. `Model Variants/Neural Networks/MLP/08_MLP_Random_Search.ipynb`
10. `Model Variants/Neural Networks/MLP/09_MLP_Genetic_Algorithm.ipynb`
11. `Model Variants/Neural Networks/MLP/10_MLP_HPO_Comparison.ipynb`
12. `Model Variants/Neural Networks/MLP/11_MLP_Final_Selected_Model.ipynb`

## MLP artifact layout

- Cache: `outputs/mlp/cache/`
- Final split manifests: `outputs/mlp/manifests/`
- Configs and selected hyperparameters: `outputs/mlp/configs/`
- Trial tables: `outputs/mlp/tables/`
- Training histories: `outputs/mlp/histories/`
- Metrics: `outputs/mlp/metrics/`
- Figures: `outputs/mlp/figures/`
- Models: `outputs/mlp/models/`
- Archived pilot notebooks: `outputs/mlp/pilot_legacy/`

## Method policy

- MLP features are `40` MFCC coefficient means plus `40` standard deviations, giving an `80-D` vector.
- `StandardScaler` is fitted on training features only.
- Class weights are calculated from `y_train` only.
- Controlled trials vary one factor at a time and use validation macro-F1 as the primary selection metric.
- Random Search and Genetic Algorithm use the same evidence-informed domain and the same `40` unique-evaluation budget.
- Multi-seed confirmation uses `[42, 123, 2026]`.
- Only `11_MLP_Final_Selected_Model.ipynb` may compute held-out test performance.
- Refactor notebooks contain placeholders instead of fabricated results until the real runs are executed.
