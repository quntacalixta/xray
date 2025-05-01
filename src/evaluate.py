import torch
import numpy as np
from src.models.model import ChestXRayClassifier
from src.data.preprocessing import create_data_loaders
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(threshold=0.5, model_path='best_model.pth'):
    """Enhanced evaluation with threshold adjustment and error analysis"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayClassifier.load_from_checkpoint(model_path, device)
    
    # Load test data
    _, _, test_loader = create_data_loaders('data/raw/chest_xray')
    
    # Evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # CHANGED: Threshold adjustment
            preds = (probabilities[:, 1] > threshold).long()
            
            all_probs.extend(probabilities.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # NEW: Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # NEW: Classification report with adjusted threshold
    print(f"\nClassification Report (Threshold={threshold}):")
    print(classification_report(
        all_labels, all_preds,
        target_names=['NORMAL', 'PNEUMONIA']
    ))
    
    # NEW: Return values for further analysis
    return {
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs)
    }
def calculate_clinical_metrics(y_true, y_pred, y_prob):
    """Calculate clinical metrics like PPV and NPV"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Métriques cliniques
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Valeur prédictive positive
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Valeur prédictive négative
    
    clinical_metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv
    }
    
    print("\nClinical Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value (PPV): {ppv:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    
    return clinical_metrics

if __name__ == '__main__':
    # NEW: Test multiple thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        print(f"\n{'='*50}")
        print(f"EVALUATION WITH THRESHOLD = {threshold}")
        print(f"{'='*50}")
        evaluate(threshold)