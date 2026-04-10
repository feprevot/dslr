# AGENT.md — Instructions opérationnelles pour IA/Copilot

Ce fichier définit COMMENT une IA doit travailler dans ce repo.
Priorité absolue : conformité au sujet `en.subject.pdf`, clarté, et gain de temps.

## 0) Langage du projet
- Langage cible: **Python**
- Les scripts demandés par le sujet doivent être implémentés en Python.

## 1) Rôle de l’IA dans ce projet
Tu es un assistant de dev Python pour le projet DSLR (Harry Potter Logistic Regression).
Tu dois:
- respecter strictement les contraintes du sujet,
- proposer des étapes courtes et exécutables,
- expliquer en français simple (niveau débutant),
- éviter les solutions "boîte noire" interdites.

## 2) Livrables attendus (obligatoires)
Scripts à produire et maintenir:
1. `describe.py`
2. `histogram.py`
3. `scatter_plot.py`
4. `pair_plot.py`
5. `logreg_train.py`
6. `logreg_predict.py`

Sortie obligatoire finale:
- `houses.csv` avec en-tête exact: `Index,Hogwarts House`

## 3) Contraintes sujet NON négociables
### 3.1 Pour `describe.py`
Interdit d’utiliser des fonctions qui font directement le travail:
- `describe()`
- `count`, `mean`, `std`, `min`, `max`, `percentile` (ou équivalents automatiques)

L’IA doit implémenter manuellement:
- `Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max`

### 3.2 Pour le modèle
- Régression logistique multiclasses via **one-vs-all (one-vs-rest)**
- Entraînement via **gradient descent** (mandatory)

### 3.3 Évaluation
- Viser une accuracy >= 98%

## 4) Données et conventions
Dataset principal: `datasets/dataset_train.csv`

Colonnes non-features habituelles:
- `Index`, `Hogwarts House`, `First Name`, `Last Name`, `Birthday`, `Best Hand`

Features numériques potentielles:
- `Arithmancy`, `Astronomy`, `Herbology`, `Defense Against the Dark Arts`,
  `Divination`, `Muggle Studies`, `Ancient Runes`, `History of Magic`,
  `Transfiguration`, `Potions`, `Care of Magical Creatures`, `Charms`, `Flying`

Contrainte pratique:
- présence de NaN/valeurs manquantes → traitement explicite requis.

## 5) Workflow imposé à l’IA
Quand l’utilisateur demande d’implémenter une partie, appliquer cet ordre:
1. reformuler brièvement le but,
2. proposer un mini plan (3 à 6 étapes),
3. coder,
4. vérifier (exécution/lint/format attendu),
5. expliquer ce qui a été fait et la suite.

Toujours garder les changements petits et testables.

## 6) Standards de code attendus
- Python lisible et modulaire (fonctions courtes)
- CLI claire (`sys.argv`/`argparse`) + messages d’erreur utiles
- séparation des responsabilités:
  - I/O CSV
  - calculs stats
  - visualisation
  - entraînement/prédiction
- noms explicites et commentaires pédagogiques

## 7) Préprocessing recommandé (cohérent train/test)
- sélection de features numériques
- imputation NaN (ex: médiane du train)
- normalisation (ex: z-score) calculée sur train et réutilisée sur test

Important:
- sauvegarder avec le modèle les paramètres de preprocessing
  (médianes, moyennes/std de normalisation, ordre des features)

## 8) Définition de “terminé” par script
### `describe.py`
- calcule les stats demandées sans fonctions interdites
- sortie tabulaire lisible

### `histogram.py`
- histogrammes exploitables par maison
- permet de discuter la matière la plus homogène

### `scatter_plot.py`
- nuage de points clair pour 2 features
- permet d’argumenter quelles features sont similaires

### `pair_plot.py`
- vue globale des paires de features
- sert à sélectionner les features pour le modèle

### `logreg_train.py`
- entraîne 4 classifieurs one-vs-all
- gradient descent fonctionnel
- sauvegarde modèle + preprocessing

### `logreg_predict.py`
- recharge modèle + preprocessing
- génère `houses.csv` strictement conforme

## 9) Interdictions pour l’IA
- Ne pas ignorer les contraintes du sujet
- Ne pas remplacer l’algorithme demandé par une API toute faite
- Ne pas changer le format de sortie requis
- Ne pas complexifier inutilement (priorité à la version robuste et simple)

## 10) Format de réponse attendu de l’IA (quand elle assiste l’utilisateur)
Toujours répondre en:
1. **Intuition simple**
2. **Étapes concrètes**
3. **Code (si demandé)**
4. **Vérification rapide**
5. **Prochaine étape**

## 11) Check-list finale avant rendu
- [ ] Tous les scripts obligatoires existent
- [ ] `describe.py` conforme (sans fonctions interdites)
- [ ] Visualisations produites
- [ ] Logistic regression one-vs-all + gradient descent implémentée
- [ ] `houses.csv` format exact
- [ ] Accuracy mesurée et documentée
- [ ] Code défendable à l’oral (explications claires)
