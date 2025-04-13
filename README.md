# Chest X-Ray Pneumonia Classification

## 1. Aperçu du projet

Ce projet développe un modèle d'apprentissage profond pour classifier des images radiographiques pulmonaires comme normales ou présentant une pneumonie. Il utilise une approche par transfert d'apprentissage avec une architecture ResNet18 pré-entraînée sur ImageNet, ajustée pour cette tâche de classification binaire.

## 2. Structure du projet

```
.
├── configs/                 # Fichiers de configuration
│   └── config.yaml          # Configuration principale
├── data/                    # Répertoire de données
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données traitées
│   └── splits/              # Divisions des données
├── notebooks/               # Notebooks Jupyter
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
├── src/                     # Code source
│   ├── data/                # Traitement des données
│   │   ├── preprocessing.py # Chargement et prétraitement
│   │   ├── dataset.py       # Classe Dataset
│   │   └── class_balance.py # Analyse de l'équilibre des classes
│   ├── models/              # Architecture du modèle
│   │   └── model.py         # Définition du modèle
│   ├── utils/               # Utilitaires
│   │   └── visualization.py # Outils de visualisation
│   ├── train.py             # Script d'entraînement
│   ├── evaluate.py          # Script d'évaluation
│   └── predict.py           # Script de prédiction
├── analyze_dataset.py       # Script d'analyse du jeu de données
├── setup_check.py           # Vérification de l'environnement
├── run.py                   # Script d'exécution principal
├── requirements.txt         # Dépendances du projet
└── README.md                # Documentation du projet
```

## 3. Instructions d'installation

### 3.1 Prérequis
- Python 3.8+
- GPU compatible CUDA (recommandé)

### 3.2 Cloner le dépôt
```bash
git clone https://github.com/yourusername/xray.git
cd xray
```

### 3.3 Créer un environnement virtuel
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows:
venv\Scripts\activate
# Sur macOS/Linux:
source venv/bin/activate
```

### 3.4 Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3.5 Vérifier l'installation
```bash
python setup_check.py
```

## 4. Préparation des données

### 4.1 Télécharger le jeu de données
Téléchargez le jeu de données Chest X-Ray Pneumonia depuis Kaggle et extrayez-le dans le répertoire `data/raw/`.

La structure attendue du jeu de données est la suivante:
```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 4.2 Analyser le jeu de données
```bash
python analyze_dataset.py
```

### 4.3 Analyser l'équilibre des classes
```bash
python -m src.data.class_balance
```
Cette commande analysera la distribution des classes et suggérera des stratégies pour gérer tout déséquilibre potentiel.

## 5. Utilisation

### 5.1 Script d'exécution principal
Le script `run.py` sert de point d'entrée principal pour exécuter toutes les étapes du projet :

```bash
# Afficher l'aide
python run.py --help

# Exécuter toutes les étapes (configuration, analyse, entraînement, évaluation, visualisation)
python run.py --all

# Exécuter uniquement certaines étapes
python run.py --setup --analyze
python run.py --train --config configs/config.yaml
python run.py --evaluate --thresholds 0.3 0.5 0.7
python run.py --visualize
```

### 5.2 Entraînement du modèle
```bash
# Utiliser le script d'exécution principal
python run.py --train

# Ou directement avec le script d'entraînement
python -m src.train
```

L'entraînement utilisera les paramètres définis dans `configs/config.yaml` et les résultats seront suivis avec Weights & Biases (wandb).

### 5.3 Évaluation du modèle
```bash
# Utiliser le script d'exécution principal
python run.py --evaluate

# Ou directement avec le script d'évaluation
python -m src.evaluate

# Tester plusieurs seuils de classification
python -m src.evaluate --threshold 0.3
python -m src.evaluate --threshold 0.5
python -m src.evaluate --threshold 0.7
```

### 5.4 Prédiction sur de nouvelles images
```bash
# Prédiction sur une image unique
python -m src.predict --image path/to/image.jpg --output results/

# Prédiction par lots sur un dossier d'images
python -m src.predict --batch path/to/folder/ --output results/

# Utilisation d'un modèle spécifique et d'un seuil personnalisé
python -m src.predict --image path/to/image.jpg --model best_model.pth --threshold 0.4 --cuda
```

### 5.5 Exploration avec les notebooks
```bash
jupyter notebook
```
Accédez au répertoire `notebooks/` pour explorer :
- `01_data_exploration.ipynb` : Analyse du jeu de données
- `02_model_development.ipynb` : Développement du modèle et réglage des hyperparamètres
- `03_results_analysis.ipynb` : Analyse détaillée des performances du modèle

## 6. Configuration

Le modèle et les paramètres d'entraînement peuvent être personnalisés dans `configs/config.yaml` :

```yaml
# Configuration du modèle
model:
  name: "resnet18"  # Autres options: "resnet34", "resnet50", etc.
  pretrained: true
  num_classes: 2

# Hyperparamètres d'entraînement
training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  optimizer: "adam"  # Autres options: "sgd"
  weight_decay: 0.00001

# Prétraitement des données
preprocessing:
  image_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# Augmentation des données
augmentation:
  horizontal_flip: true
  rotation_range: 20
  brightness_range: [0.8, 1.2]

# Journalisation et suivi
logging:
  project_name: "chest-xray-pneumonia"
  experiment_name: "baseline-resnet18"
```

## 7. Métriques de performance

Le modèle est évalué à l'aide de :
- Précision (Accuracy)
- Précision (Precision)
- Rappel (Recall)
- Score F1
- Matrice de confusion

Les visualisations des résultats incluent :
- Distribution des prédictions correctes et incorrectes
- Exemples de faux positifs et faux négatifs
- Optimisation du seuil de classification
- Courbes ROC et Precision-Recall

## 8. Améliorations futures

- Expérimenter avec différentes architectures (ResNet34, EfficientNet)
- Mettre en œuvre la validation croisée
- Améliorer les techniques d'augmentation des données
- Résoudre le déséquilibre des classes
- Ajouter l'explicabilité avec des visualisations Grad-CAM
- Déployer le modèle en tant que service web

## 9. Dépannage

### 9.1 Problèmes courants
- **Erreur CUDA out of memory** : Réduisez la taille du lot (batch_size) dans la configuration
- **Erreur ModuleNotFoundError** : Assurez-vous que l'environnement virtuel est activé et que toutes les dépendances sont installées
- **Performances faibles sur l'ensemble de validation** : Ajustez les hyperparamètres, en particulier le taux d'apprentissage (learning_rate)

### 9.2 Vérification du système
Pour diagnostiquer les problèmes potentiels, exécutez :
```bash
python setup_check.py
```

## 10. Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 11. Contact

Pour toute question ou suggestion, veuillez contacter [votre-email@example.com].
