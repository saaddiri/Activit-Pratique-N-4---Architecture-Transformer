# Architecture Transformer

Activité Pratique N°4 — Pr. Mohamed YOUSSFI

Du réseau récurrent au Transformer : on entraîne un **RNN**, un **LSTM** et un **Transformer** à prédire la valeur suivante d'une sinusoïde, puis on utilise deux grands modèles préentraînés, **GPT-2** (génération de texte) et **BERT** (mot masqué).

Le notebook `architecture_transformer.ipynb` reprend le code et les explications du professeur sans les modifier. Il est livré **exécuté**, avec les sorties visibles.

## Technologies

| Rôle | Outil |
|------|-------|
| Gestion de l'environnement | [uv](https://docs.astral.sh/uv/), Python 3.11 |
| Réseaux de neurones | PyTorch (CPU) |
| Modèles préentraînés | Hugging Face `transformers` : `gpt2`, `bert-base-multilingual-cased` |
| Visualisation | Matplotlib |

Tout tourne sur CPU (portable i7-6500U, sans GPU) : les modèles entraînés sont volontairement petits.

## Structure du dépôt

```
.
├── architecture_transformer.ipynb   # notebook principal (exécuté, sorties visibles)
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

- [ ] **Exercice 1** : prédiction de la consommation électrique avec un Transformer
- [ ] **Exercice 2** : mini-GPT caractère par caractère
- [ ] **Exercice 3** : mini-BERT pour retrouver des mots masqués
