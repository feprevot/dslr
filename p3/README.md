# P3 Commands

<!-- Train model on full train set -->

python p3/logreg_train.py datasets/dataset_train.csv

<!-- Predict on official test set -->

python p3/logreg_predict.py datasets/dataset_test.csv model.json

<!-- Create 80/20 validation split -->

python p3/make_validation_split.py datasets/dataset_train.csv

<!-- Train on 80% split -->

python p3/logreg_train.py datasets/validation/dataset_train_80.csv

<!-- Predict on 20% split + show evaluation report -->

python p3/logreg_predict.py datasets/validation/dataset_val_20.csv model.json datasets/validation/dataset_val_20.csv
