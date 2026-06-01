# DSLR — DataScience × Logistic Regression

A **42** project (*outer expert* branch). The goal: rebuild Hogwarts' magic
**Sorting Hat**. From the students' grades across several courses, predict their
house — Gryffindor, Hufflepuff, Ravenclaw or Slytherin — using a **multi-class
logistic regression implemented *from scratch***, without any machine-learning
library.

The project follows the subject in three stages: explore the data
(statistics), visualize it to pick the relevant features, then train and
evaluate the classifier.

---

## Table of contents

- [What is done from scratch](#what-is-done-from-scratch)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Part 1 — Statistical analysis](#part-1--statistical-analysis)
  - [Part 2 — Visualization](#part-2--visualization)
  - [Part 3 — Logistic regression](#part-3--logistic-regression)
  - [Validation (optional)](#validation-optional)
- [Technical approach](#technical-approach)
- [Selected features](#selected-features)
- [Output formats](#output-formats)
- [Results](#results)

---

## What is done from scratch

The core of the project is rewritten without any ML dependency:

- **Descriptive statistics** ([`utils.py`](utils.py)): `mean`, `std` (sample,
  `n-1`), `min`, `max`, `percentile` (linear interpolation), `median`, `range` —
  without `numpy.mean`, `pandas.describe`, etc.
- **Logistic regression** ([`p3/logreg_utils/`](p3/logreg_utils/)): sigmoid,
  binary log-loss, gradient descent and the *one-vs-all* strategy, all written by
  hand. `numpy` is used only for matrix algebra, `pandas` for reading CSVs, and
  `matplotlib`/`seaborn` for the plots.

> Note: `scikit-learn` is listed in `requirements.txt` but is never imported
> anywhere in the code — all of the learning logic is hand-written. You can drop
> it from the file if you want it to strictly reflect the implementation.

---

## Project structure

```
dslr/
├── utils.py                      # Hand-written basic statistics
├── datasets/
│   ├── dataset_train.csv         # Students WITH their house (training)
│   └── dataset_test.csv          # Students WITHOUT their house (to predict)
├── p1/
│   └── describe.py               # Hand-written equivalent of df.describe()
├── p2/
│   ├── histogram.py              # Course with the most homogeneous distribution
│   ├── scatter_plot.py           # The two most similar features
│   └── pair_plot.py              # Overview to help pick features
└── p3/
    ├── logreg_train.py           # Training    → model.json
    ├── logreg_predict.py         # Prediction  → houses.csv
    ├── make_validation_split.py  # Stratified 80/20 split
    ├── logreg_utils/
    │   ├── binary_logreg.py      # sigmoid, log-loss, gradient descent
    │   ├── preprocessing.py      # feature selection, imputation, z-score
    │   └── io_utils.py           # CSV loading + validation
    ├── reports/
    │   ├── predict_report.py     # prediction summary
    │   └── evaluation_report.py  # accuracy, confusion matrix, P/R/F1
    ├── README.md                 # Command cheat-sheet
    └── WORKFLOW.md               # Detailed workflow + Mermaid diagrams
```

---

## Setup

Python 3.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn` (and `scikit-learn`,
listed but unused — see above).

---

## Usage

> Run the commands below **from the repository root**: the scripts in `p3/`
> import `utils.py` and the `logreg_utils/` modules relative to that root.

### Part 1 — Statistical analysis

```bash
python p1/describe.py datasets/dataset_train.csv
# Optional: limit the number of displayed columns
python p1/describe.py datasets/dataset_train.csv 5
```

For each numeric column it prints: `Missing`, `Missing%`, `Count`, `Mean`,
`Std`, `Min`, `25%`, `50%`, `75%`, `Max`, `Range` — all computed by hand.

### Part 2 — Visualization

These three scripts read `datasets/dataset_train.csv` directly (no arguments to
pass) and open a `matplotlib` window.

```bash
python p2/histogram.py     # Detects and plots the most homogeneous course across houses
python p2/scatter_plot.py  # Plots the two most similar features
python p2/pair_plot.py     # Pair plot of the selected features, colored by house
```

- `histogram.py` automatically picks the course whose spread across houses is
  the smallest (lowest mean of per-house standard deviations).
- `scatter_plot.py` plots `Astronomy` vs `Defense Against the Dark Arts`, the two
  near-identical (strongly correlated) courses.

### Part 3 — Logistic regression

Train the model on the training set:

```bash
python p3/logreg_train.py datasets/dataset_train.csv
# → writes model.json (weights, bias and preprocessing parameters)
```

Predict houses on the official test set:

```bash
python p3/logreg_predict.py datasets/dataset_test.csv model.json
# → writes houses.csv
```

### Validation (optional)

To measure performance, create a stratified 80/20 split from the training set
(which does contain the true houses):

```bash
# 1. Create the split (seed 42 by default, overridable: ... dataset_train.csv 7)
python p3/make_validation_split.py datasets/dataset_train.csv
#    → datasets/validation/dataset_train_80.csv
#    → datasets/validation/dataset_val_20.csv

# 2. Train on the 80%
python p3/logreg_train.py datasets/validation/dataset_train_80.csv

# 3. Predict on the 20%, passing the ground truth as the 3rd argument
python p3/logreg_predict.py datasets/validation/dataset_val_20.csv model.json \
                            datasets/validation/dataset_val_20.csv
```

When a 3rd argument (ground truth) is provided, `logreg_predict.py` prints an
**evaluation report**: overall accuracy, confusion matrix and, per house,
precision / recall / F1 / support.

> Reminder: **precision** answers "when the model says *Gryffindor*, is it
> right?", **recall** answers "does it find all the *Gryffindors*?".

---

## Technical approach

The full details, with diagrams, live in [`p3/WORKFLOW.md`](p3/WORKFLOW.md). In
short:

1. **Preprocessing** ([`preprocessing.py`](p3/logreg_utils/preprocessing.py))
   - select the 10 relevant course features;
   - impute `NaN` with the **median** (robust to outliers);
   - **z-score standardization** `x' = (x − μ) / σ`.

   Medians, `μ` and `σ` are **computed on the train set only**, then stored in
   `model.json` and re-applied identically at prediction time. This is critical:
   the learned weights are tuned for a specific scale; recomputing those
   statistics on the test set would distort the score `z = X·w + b`.

2. **One-vs-all** — since logistic regression is binary, we train **4
   independent classifiers** ("this house against all the others"). At
   prediction time we compute the 4 probabilities and take the `argmax`.

3. **Gradient descent**
   ([`binary_logreg.py`](p3/logreg_utils/binary_logreg.py)) — for each
   classifier, weights and bias initialized to zero, then at every epoch:
   `z = X·w + b`, `p = sigmoid(z)`, log-loss computation and the update
   `w ← w − lr·∂loss/∂w`. Hyperparameters: `learning_rate = 0.1`,
   `epochs = 1000`.

---

## Selected features

Out of the 13 courses, **10** are kept for training:

```
Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes,
History of Magic, Transfiguration, Potions, Charms, Flying
```

Dropped (justified in the code): **Arithmancy** and **Care of Magical
Creatures** (spread homogeneously across houses, weakly discriminative), and
**Defense Against the Dark Arts** (nearly identical to *Astronomy*, hence
redundant).

---

## Output formats

- **`model.json`**: `features` (order matters), `houses`, `weights` and `bias`
  per house, plus `train_medians`, `train_means`, `train_stds`.
- **`houses.csv`**: the exact format required by the subject.

  ```
  Index,Hogwarts House
  0,Gryffindor
  1,Hufflepuff
  ...
  ```

---

## Results

Accuracy ≥ **98%** on the 80/20 validation split and on the subject's official
test set.
