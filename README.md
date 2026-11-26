# Architecture Transformer

Activité Pratique N°4

Du réseau récurrent au Transformer : on entraîne un **RNN**, un **LSTM** et un **Transformer** à prédire la valeur suivante d'une sinusoïde, puis on utilise deux grands modèles préentraînés, **GPT-2** (génération de texte) et **BERT** (mot masqué). Les exercices construisent ensuite trois Transformers depuis zéro : prévision de consommation électrique, mini-GPT et mini-BERT.

Le notebook `architecture_transformer.ipynb` reprend le code et les explications du cours sans les modifier ; les solutions des exercices sont ajoutées dans de nouvelles cellules, sous chaque énoncé. Il est livré **exécuté**, avec les sorties visibles.

## Technologies

| Rôle | Outil |
|------|-------|
| Gestion de l'environnement | [uv](https://docs.astral.sh/uv/), Python 3.11 |
| Réseaux de neurones | PyTorch (CPU) |
| Modèles préentraînés | Hugging Face `transformers` : `gpt2`, `bert-base-multilingual-cased` |
| Données | pandas, NumPy |
| Visualisation | Matplotlib |

Tout tourne sur CPU (portable i7-6500U, sans GPU) : les modèles entraînés sont volontairement petits.

## Structure du dépôt

```
.
├── architecture_transformer.ipynb   # notebook principal (exécuté, sorties visibles)
├── corpus.txt                       # corpus de l'exercice 2 (~37 000 caractères)
├── pyproject.toml / uv.lock         # dépendances figées
└── README.md
```

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/saaddiri/Activit-Pratique-N-4---Architecture-Transformer.git
cd Activit-Pratique-N-4---Architecture-Transformer
uv sync
```

Pour l'exercice 1, téléchargez le jeu de données [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (UCI) et extrayez `household_power_consumption.txt` (127 Mo) à côté du notebook. Il n'est pas versionné.

Ouvrez ensuite `architecture_transformer.ipynb` dans VS Code (ou Jupyter), sélectionnez le noyau `.venv` et exécutez les cellules dans l'ordre. Au premier lancement, les poids de GPT-2 (~550 Mo) et de BERT multilingue (~700 Mo) sont téléchargés automatiquement.

## Avancement

### Première Partie

- [x] **Série temporelle** : `y = sin(x)` sur 200 points, fenêtres glissantes de 10 valeurs → valeur suivante (150 exemples d'entraînement, 40 de test, découpage chronologique)
- [x] **RNN** : `nn.RNN` (16 unités) + `Linear(16, 1)`, 300 époques, Adam
- [x] **LSTM** : `nn.LSTM` (16 unités, portes d'oubli / d'entrée / de sortie) + `Linear(16, 1)`
- [x] **Transformer** : projection `1 → 16`, positions apprises, encodeur à 2 têtes, sortie sur la dernière position
- [x] **GPT** : GPT-2 préentraîné, génération autorégressive de la suite d'un texte
- [x] **BERT** : BERT multilingue préentraîné, top-5 des mots pour un `[MASK]`

Résultats obtenus sur le test (MSE, sinusoïde) :

| Modèle | Erreur de test (MSE) |
|---|---|
| RNN | 0,000028 |
| LSTM | 0,000003 |
| Transformer | 0,000006 |

Les trois modèles suivent la sinusoïde presque parfaitement ; le LSTM et le Transformer font environ 5 à 10 fois mieux que le RNN simple.

### Deuxième Partie

- [x] **Exercice 1 — Consommation électrique** : `EnergyTransformer` (projection `1 → 32`, positions apprises, 2 couches d'encodeur à 4 têtes), 24 h → heure suivante, 20 époques au maximum avec conservation du meilleur modèle sur la validation, MAE/RMSE en kW, comparaison à deux références, tracé sur 7 jours, essai avec une fenêtre de 168 h
- [x] **Exercice 2 — Mini-GPT** : blocs pré-normalisés (attention causale 4 têtes + MLP `64 → 256 → 64` GELU), tokenisation par caractères, génération de 200 caractères à 3 températures, vérification de la causalité, démonstration de l'entraînement sans masque
- [x] **Exercice 3 — Mini-BERT** : vocabulaire par mots, séquences de 32 tokens, masquage 15 % (80 / 10 / 10), encodeur bidirectionnel avec masque de padding, perte sur les positions sélectionnées, exactitude et top-3, vérification du padding
- [x] Réponses argumentées aux questions de réflexion des trois exercices

Exercice 1 — test (4 932 heures) :

| Modèle | MAE (kW) | RMSE (kW) |
|---|---|---|
| **EnergyTransformer** | **0,354** | **0,500** |
| Persistance (heure précédente) | 0,389 | 0,588 |
| Référence journalière (24 h avant) | 0,513 | 0,764 |

La fenêtre de 168 h (3 époques seulement, une époque coûte ~17 fois plus) donne une MAE de 0,351 kW contre 0,350 kW pour 24 h sur les mêmes heures : pas de gain visible.

Exercice 2 : perte de validation de 1,59 (hasard : 4,60) ; la causalité est vérifiée (modifier un caractère futur ne change pas les logits passés). Sans masque causal, la perte tombe à 0,06 mais le modèle ne sait plus générer.

Exercice 3 : sur 683 tokens sélectionnés de phrases jamais vues, **81,1 %** d'exactitude (top-3 : **94,3 %**). Exemples : « le [MASK] aboie le soir » → *chien* (96 %), « [MASK] est la capitale du maroc » → *rabat* (65 %). Changer les tokens de padding ne modifie pas les prédictions.

## Choix et adaptations

- **Corpus de l'exercice 2** : `corpus.txt` reprend les explications de la Première Partie du notebook (sans formules ni tableaux), soit environ une quinzaine de pages en français.
- **Corpus de l'exercice 3** : 3 121 phrases générées à partir de gabarits (animaux, métiers, capitales, saisons) pour que le contexte, à gauche ou à droite, permette réellement de retrouver le mot masqué ; 85 % pour l'entraînement, 15 % pour la validation.
- **Sur-apprentissage du mini-GPT** : le corpus étant petit, la perte de validation remontait après ~1 000 pas. Ajout d'un dropout de 0,1 et conservation du meilleur pas sur la validation.
- **Fenêtre de 168 h** : limitée à 3 époques à cause du coût de l'attention en O(T²) sur CPU ; la comparaison reste indicative.
