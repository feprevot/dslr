# P3 Code Structure

This folder is split by responsibility:

- `io_utils.py`
  - CSV/file loading helpers
  - `load_csv()`

- `preprocessing.py`
  - Feature/target split
  - NaN imputation
  - Standardization

- `binary_logreg.py`
  - Binary logistic regression math
  - `sigmoid()`, `binary_log_loss()`, `train_binary_logreg_gd()`

- `logreg_train.py`
  - Training script entry point
  - Calls I/O + preprocessing + binary training

- `logreg_predict.py`
  - Prediction script entry point

- `data_utils.py`
  - Compatibility re-export module (legacy imports)

## Data flow in train

1. Load CSV
2. Split into `X` (numeric features) and `y` (house labels)
3. Impute NaN with train medians
4. Standardize with train means/stds
5. Train first binary model (Gryffindor vs all)
