# Plan d’action — Partie 3 (Logistic Regression) pour débutant Python

## Objectif de ta partie
Tu dois livrer 2 scripts :
- `logreg_train.py`
- `logreg_predict.py`

Contraintes sujet :
- Régression logistique **one-vs-all** (4 maisons)
- Entraînement avec **gradient descent**
- Sortie finale `houses.csv` au format exact :
  - `Index,Hogwarts House`

---

## 1) Ce qu’il faut récupérer de la partie 1/2 (ton pote)
Pour ne pas coder à l’aveugle, demande :
1. la liste des features retenues (matières gardées),
2. la stratégie NaN (ex: imputation médiane),
3. la normalisation utilisée (si oui, laquelle).

---

## 2) Plan de travail concret (6 étapes)

### Étape B — ML minimum vital (1/2 journée)
Comprendre :
- score linéaire : $z = w \cdot x + b$  note totale (score pour une maison, grand positif, proche de 0 ou grand negatif)
- sigmoïde : $\sigma(z)=\frac{1}{1+e^{-z}}$  transforme le score brut en proba entre 0 et 1
- log-loss, mesure de l'erreur, a quel point le resultat reel est loin de notre prevision
- gradient descent (petits ajustements de `w`, `b`), comment corriger l'erreur (quelle direction, quel est le pas)
- one-vs-all (4 classifieurs binaires).

### Étape C — Créer le squelette des scripts (1/2 journée)
Dans `logreg_train.py` :
- chargement dataset,
- sélection features,
- imputation NaN,
- normalisation,
- entraînement binaire,
- boucle one-vs-all,
- sauvegarde modèle + preprocessing.

Dans `logreg_predict.py` :
- chargement modèle,
- même preprocessing,
- calcul scores des 4 classes,
- `argmax`,
- génération `houses.csv`.

### Étape D — Implémenter l’entraînement (1 à 2 jours)
Ordre :
1. logistic binaire (une maison vs reste),
2. vérifier que la loss descend,
3. généraliser en one-vs-all,
4. sauvegarder poids + biais + ordre des features + paramètres preprocessing.

### Étape E — Implémenter la prédiction (1 jour)
- recharger exactement les mêmes paramètres,
- prédire sur test,
- produire `houses.csv` strictement conforme.

### Étape F — Validation finale (1 jour)
Checklist :
- loss stable (et plutôt descendante),
- pas de fuite train/test,
- format de sortie exact,
- code explicable facilement à l’oral.

---

## 3) Notions à comprendre absolument
1. différence `train` vs `predict`,
2. importance d’utiliser le même preprocessing,
3. rôle des poids,
4. pourquoi normaliser,
5. pourquoi one-vs-all pour 4 classes.

---

## 4) Pièges à éviter
- normaliser le test avec des stats du test,
- ignorer les NaN,
- changer l’ordre des features entre train et predict,
- mauvais format de `houses.csv`,
- learning rate trop grand.

---

## 5) Définition de réussite
C’est réussi si :
- `logreg_train.py` entraîne bien 4 modèles one-vs-all,
- `logreg_predict.py` génère un `houses.csv` valide,
- tu sais expliquer simplement : sigmoïde, loss, gradient descent, one-vs-all.

---

## Bonus pratique : mini planning de démarrage
- Jour 1 matin : Python/Numpy bases
- Jour 1 après-midi : squelette `train/predict`
- Jour 2 : entraînement binaire + loss
- Jour 3 : one-vs-all + sauvegarde modèle
- Jour 4 : predict + `houses.csv`
- Jour 5 : validation, nettoyage, explications soutenance
