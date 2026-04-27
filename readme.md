# DSLR — DataScience x Logistic Regression

Projet de la branche **outer expert** du cursus 42. L'objectif est de recréer
le Choixpeau magique de Poudlard : à partir des notes des élèves dans 13 cours,
prédire leur maison (Gryffondor, Poufsouffle, Serdaigle, Serpentard) avec une
**régression logistique multi-classes** implémentée **from scratch** — sans
scikit-learn ni aucune librairie de machine learning.

## Stack

- **Python 3** — `numpy`, `pandas`, `matplotlib`, `seaborn`
- Aucune dépendance ML : sigmoïde, log-loss, descente de gradient et stratégie
  one-vs-all sont écrites à la main.
- Statistiques descriptives (`mean`, `std`, `percentile`, …) réimplémentées
  sans `numpy.mean` ni `pandas.describe`.

## Structure du dépôt

| Dossier        | Rôle                                                              |
|----------------|-------------------------------------------------------------------|
| [p1/](p1/)     | `describe.py` — équivalent maison de `df.describe()`              |
| [p2/](p2/)     | Visualisations : histogramme, scatter plot, pair plot             |
| [p3/](p3/)     | Régression logistique : entraînement, prédiction, modules         |
| [validation/](validation/) | Split 80/20 + matrice de confusion / rapport métriques |
| [datasets/](datasets/)     | Jeux de données fournis (train / test)                 |
| [utils.py](utils.py)       | Fonctions statistiques de base                         |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Partie 1 — Analyse statistique

```bash
python p1/describe.py datasets/dataset_train.csv
```

Affiche, pour chaque feature numérique : `count`, `mean`, `std`, `min`, quartiles,
`max`, `range` — implémentés à la main.

### Partie 2 — Visualisation

```bash
python p2/histogram.py     # cours dont la distribution est la plus homogène
python p2/scatter_plot.py  # paire de features les plus corrélées
python p2/pair_plot.py     # vue d'ensemble pour choisir les features
```

### Partie 3 — Régression logistique

Entraîner le modèle :

```bash
python p3/logreg_train.py datasets/dataset_train.csv
# → produit model.json
```

Prédire sur le jeu de test :

```bash
python p3/logreg_predict.py datasets/dataset_test.csv model.json
# → produit houses.csv
```

### Validation (optionnel)

```bash
python validation/make_validation_split.py datasets/dataset_train.csv
python p3/logreg_train.py   datasets/validation/dataset_train_80.csv
python p3/logreg_predict.py datasets/validation/dataset_val_20.csv model.json \
                            datasets/validation/dataset_val_20.csv
```

Ce dernier appel imprime un rapport d'évaluation (accuracy, matrice de
confusion, précision / rappel par maison).

## Approche technique

1. **Preprocessing** — sélection des features numériques, imputation des `NaN`
   par la médiane d'entraînement, standardisation z-score. Les statistiques
   (`médianes`, `μ`, `σ`) sont **calculées sur le train uniquement** et stockées
   dans `model.json` pour être réappliquées à l'identique en prédiction (sinon
   les poids appris ne sont plus à la bonne échelle).
2. **One-vs-all** — la régression logistique étant binaire, on entraîne 4
   classifieurs indépendants (« cette maison contre toutes les autres ») et on
   prend l'`argmax` des probabilités à la prédiction.
3. **Descente de gradient** — pour chaque classifieur : `z = X·w + b`,
   `p = sigmoid(z)`, mise à jour des poids par le gradient de la log-loss
   (`learning_rate = 0.1`, `epochs = 1000`).

Détails et diagrammes du flux dans [p3/WORKFLOW.md](p3/WORKFLOW.md).

## Résultats

Accuracy ≥ 98 % sur le split de validation 80/20 et sur le jeu de test officiel
du sujet.

## Auteur

Quentin, Felix — étudiants à [42](https://42.fr).
