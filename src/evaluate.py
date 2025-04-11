import torch
import numpy as np
from src.models.model import ChestXRayClassifier
from src.data.preprocessing import create_data_loaders
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(threshold=0.5):
    """Enhanced evaluation with threshold adjustment and error analysis"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayClassifier.load_from_checkpoint('best_model.pth', device)
    
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

if __name__ == '__main__':
    # NEW: Test multiple thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        print(f"\n{'='*50}")
        print(f"EVALUATION WITH THRESHOLD = {threshold}")
        print(f"{'='*50}")
        evaluate(threshold)