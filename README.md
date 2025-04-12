# Chest X-Ray Pneumonia Classification

## Project Overview
This project develops a deep learning model to classify chest X-ray images as normal or pneumonia-positive. It uses a transfer learning approach with a ResNet18 architecture pretrained on ImageNet, fine-tuned for the binary classification task.

## Dataset
The project uses the Chest X-Ray Pneumonia dataset, which should be organized as follows:
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

## Project Structure
```
.
├── configs/                 # Configuration files
│   └── config.yaml          # Main configuration
├── data/                    # Data directory
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── splits/              # Data splits
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
├── src/                     # Source code
│   ├── data/                # Data processing
│   │   ├── preprocessing.py # Data loading and preprocessing
│   │   └── dataset.py       # Dataset class
│   ├── models/              # Model architecture
│   │   └── model.py         # Model definition
│   ├── utils/               # Utilities
│   │   └── visualization.py # Visualization tools
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── analyze_dataset.py       # Dataset analysis script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/xray.git
cd xray
```

### 2. Create a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the Chest X-Ray Pneumonia dataset from Kaggle and extract it to the `data/raw/` directory.

### 5. Analyze the Dataset
```bash
python analyze_dataset.py
```
This will create the necessary directory structure and generate dataset statistics and sample visualizations.

### 6. Train the Model
```bash
python -m src.train
```
This will train the model with the configuration specified in `configs/config.yaml`. The training progress will be tracked with Weights & Biases (wandb).

### 7. Evaluate the Model
```bash
python -m src.evaluate
```
This will evaluate the model on the test set and generate performance metrics.

### 8. Explore with Notebooks
Jupyter notebooks are provided for exploration:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory to access:
- `01_data_exploration.ipynb`: Dataset analysis
- `02_model_development.ipynb`: Model development and hyperparameter tuning
- `03_results_analysis.ipynb`: Detailed analysis of model performance

## Configuration
The model and training parameters can be customized in `configs/config.yaml`:

```yaml
# Model Configuration
model:
  name: "resnet18"
  pretrained: true
  num_classes: 2

# Training Hyperparameters
training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  optimizer: "adam"
  weight_decay: 0.00001

# Data Preprocessing
preprocessing:
  image_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Future Improvements
- Experiment with different model architectures (ResNet34, EfficientNet)
- Implement cross-validation
- Enhance data augmentation techniques
- Address class imbalance
- Add explainability with Grad-CAM visualizations
- Deploy model as a web service

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Other dependencies listed in requirements.txt