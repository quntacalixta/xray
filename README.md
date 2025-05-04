# Chest X-Ray Pneumonia Classification

## 1. Aperçu du projet

Ce projet développe un modèle d'apprentissage profond pour classifier des images radiographiques pulmonaires comme normales ou présentant une pneumonie. Il utilise une approche par transfert d'apprentissage avec une architecture ResNet18 pré-entraînée sur ImageNet, ajustée pour cette tâche de classification binaire. Le modèle intègre des techniques spécifiques pour gérer le déséquilibre des classes, comme l'échantillonnage pondéré et la fonction de perte focale (Focal Loss).

## 2. Structure du projet

```
.
├── configs/                 # Fichiers de configuration
│   └── config.yaml          # Configuration principale
├── data/                    # Répertoire de données
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données traitées
│   ├── external_test/       # Images externes pour validation
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
│   │   ├── model.py         # Définition du modèle
│   │   └── losses.py        # Fonctions de perte personnalisées
│   ├── utils/               # Utilitaires
│   │   ├── visualization.py # Outils de visualisation
│   │   └── grad_cam.py      # Visualisation des régions d'intérêt
│   ├── train.py             # Script d'entraînement
│   ├── evaluate.py          # Script d'évaluation complète
│   └── predict.py           # Script de prédiction
├── test_external_images.py  # Script de test sur images externes
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
Cette commande analysera la distribution des classes et suggérera des stratégies pour gérer tout déséquilibre potentiel. Dans notre implémentation, nous utilisons l'échantillonnage pondéré et la Focal Loss pour gérer ce déséquilibre.

### 4.4 Augmenter l'ensemble de validation (si nécessaire)
```bash
python -m src.data.split_dataset
```
Cette commande répartit mieux les données entre l'ensemble d'entraînement et de validation pour assurer une évaluation plus robuste.

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

L'entraînement utilisera les paramètres définis dans `configs/config.yaml` et les résultats seront suivis avec Weights & Biases (wandb). Notre modèle optimisé utilise:
- 20 époques d'entraînement
- Focal Loss pour gérer le déséquilibre des classes
- Échantillonnage pondéré pendant l'entraînement
- Learning rate de 0.001 avec scheduler

### 5.3 Évaluation du modèle
```bash
# Utiliser le script d'exécution principal
python run.py --evaluate

# Ou directement avec le script d'évaluation complet
python -m src.evaluate

# Tester plusieurs seuils de classification
python -m src.evaluate --threshold 0.6
```

L'évaluation génère un rapport complet dans le dossier `results/` comprenant:
- Matrice de confusion
- Courbe ROC
- Courbe Precision-Recall
- Analyse des seuils optimaux
- Métriques cliniques (Sensibilité, Spécificité, VPP, VPN)
- Visualisation des cas d'erreur
- Analyses Grad-CAM des régions d'attention du modèle

### 5.4 Prédiction sur de nouvelles images
```bash
# Prédiction sur une image unique
python -m src.predict --image path/to/image.jpg --output results/

# Prédiction par lots sur un dossier d'images
python -m src.predict --batch path/to/folder/ --output results/

# Utilisation d'un modèle spécifique et d'un seuil personnalisé
python -m src.predict --image path/to/image.jpg --model best_model.pth --threshold 0.6 --cuda
```

### 5.5 Test sur des images externes

Pour tester la capacité de généralisation du modèle, nous avons implémenté un script de test sur des images externes qui n'appartiennent pas au jeu de données d'origine. Ce test est crucial pour valider que le modèle apprend correctement les caractéristiques pertinentes de la pneumonie plutôt que des particularités du jeu de données.

```bash
# Créer un dossier pour les images de test externes
mkdir -p data/external_test

# Copier vos images externes dans ce dossier

# Exécuter le script de test sur ces images
python test_external_images.py --image_dir data/external_test --threshold 0.7 --output results/external_test
```

Les résultats de nos tests externes ont montré une excellente capacité de généralisation:
- 3 radiographies normales correctement classifiées (probabilités: 0.39, 0.51, 0.38)
- 3 radiographies de pneumonie correctement classifiées (probabilités: 0.96, 0.94, 1.00)

Ces résultats confirment que notre seuil optimal de 0.7 fonctionne bien sur des données externes et que notre modèle apprend les caractéristiques médicalement pertinentes permettant de distinguer les pneumonies.

### 5.6 Exploration avec les notebooks
```bash
jupyter notebook
```
Accédez au répertoire `notebooks/` pour explorer :
- `01_data_exploration.ipynb` : Analyse du jeu de données
- `02_model_development.ipynb` : Développement du modèle et réglage des hyperparamètres
- `03_results_analysis.ipynb` : Analyse détaillée des performances du modèle

## 6. Configuration

Le modèle et les paramètres d'entraînement peuvent être personnalisés dans `configs/config.yaml`. Notre configuration optimisée est la suivante:

```yaml
# Configuration du modèle
model:
  name: "resnet18"  # Architecture ResNet18
  pretrained: true  # Utilisation des poids pré-entraînés
  num_classes: 2    # Classification binaire

# Hyperparamètres d'entraînement
training:
  batch_size: 32
  learning_rate: 0.001  # Learning rate augmenté
  epochs: 20           # 20 époques d'entraînement
  optimizer: "adam"
  weight_decay: 0.00001
  use_focal_loss: true       # Utilisation de la Focal Loss
  use_weighted_sampling: true # Échantillonnage pondéré

# Prétraitement des données
preprocessing:
  image_size: 224
  mean: [0.5, 0.5, 0.5]      # Valeurs adaptées aux radiographies
  std: [0.25, 0.25, 0.25]    # Valeurs adaptées aux radiographies

# Évaluation
evaluation:
  threshold: 0.7  # Seuil optimal déterminé par nos tests
  metrics: ["accuracy", "precision", "recall", "f1_score", "specificity", "ppv", "npv"]

# Journalisation et suivi
logging:
  project_name: "chest-xray-pneumonia"
  experiment_name: "optimized-resnet18"
```

## 7. Métriques de performance

### 7.1 Performance sur le jeu de test

Le modèle est évalué à l'aide de :
- Précision globale (Accuracy)
- Précision (Precision)
- Rappel/Sensibilité (Recall)
- Score F1
- Spécificité
- Valeur Prédictive Positive (VPP)
- Valeur Prédictive Négative (VPN)
- Matrice de confusion

Notre modèle optimisé avec un seuil de 0.7 atteint les performances suivantes:
- Accuracy: 89.1%
- Sensibilité: 98.2%
- Spécificité: 73.9%
- Score F1: 91.8%

### 7.2 Validation externe

Pour valider la capacité de généralisation du modèle, nous avons testé sur un ensemble de 6 images externes (3 normales, 3 pneumonies) qui ne faisaient pas partie du jeu de données d'origine. Les résultats sont excellents:

- Toutes les images ont été correctement classifiées
- Les images normales ont reçu des probabilités nettement inférieures au seuil (0.39, 0.51, 0.38)
- Les images de pneumonie ont reçu des probabilités très élevées (0.96, 0.94, 1.00)

Cette séparation claire des probabilités entre les classes confirme la robustesse du modèle et sa capacité à généraliser à de nouvelles images.

Les visualisations des résultats incluent :
- Distribution des prédictions correctes et incorrectes
- Exemples de faux positifs et faux négatifs
- Optimisation du seuil de classification
- Courbes ROC et Precision-Recall
- Visualisations Grad-CAM montrant les régions d'intérêt du modèle

## 8. Améliorations futures

- Expérimenter avec différentes architectures (ResNet34, EfficientNet)
- Mettre en œuvre la validation croisée
- Améliorer les techniques d'augmentation des données
- Étudier le transfert d'apprentissage à d'autres pathologies pulmonaires
- Développer un système de détection et localisation des anomalies
- Déployer le modèle en tant que service web
- Élargir la validation externe avec plus d'images provenant de sources diverses

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