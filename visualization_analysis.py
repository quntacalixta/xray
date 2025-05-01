import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from src.models.model import ChestXRayClassifier
from src.data.preprocessing import create_data_loaders
from src.utils.grad_cam import visualize_gradcam

def analyze_model_predictions(threshold=0.6):
    """Analyze model predictions with examples and Grad-CAM"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayClassifier.load_from_checkpoint('best_model.pth', device=device)
    model.eval()
    
    # Get test data
    _, _, test_loader = create_data_loaders(
        data_dir='data/raw/chest_xray',
        batch_size=64,
        use_weighted_sampling=False
    )
    
    # Create output directory
    os.makedirs('analysis_results', exist_ok=True)
    
    # Collect predictions
    all_images = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            preds = (probs[:, 1] > threshold).long()
            
            all_images.extend(images.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Identify different prediction categories
    correct_normal = (all_labels == 0) & (all_preds == 0)
    correct_pneumonia = (all_labels == 1) & (all_preds == 1)
    false_positives = (all_labels == 0) & (all_preds == 1)
    false_negatives = (all_labels == 1) & (all_preds == 0)
    
    # Visualize probability distributions
    plt.figure(figsize=(10, 6))
    plt.hist(all_probs[all_labels == 0], bins=20, alpha=0.5, label='NORMAL')
    plt.hist(all_probs[all_labels == 1], bins=20, alpha=0.5, label='PNEUMONIA')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Probability Distributions by Class')
    plt.xlabel('Predicted Probability of Pneumonia')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis_results/probability_distributions.png', dpi=300, bbox_inches='tight')
    
    # Display sample predictions with Grad-CAM
    categories = [
        ('Correctly Classified Normal', correct_normal, 3),
        ('Correctly Classified Pneumonia', correct_pneumonia, 3),
        ('False Positives', false_positives, 3),
        ('False Negatives', false_negatives, 3)
    ]
    
    # Get image paths from test folder for Grad-CAM
    img_paths = []
    for cat_name, cat_mask, n_samples in categories:
        if sum(cat_mask) > 0:
            indices = np.where(cat_mask)[0]
            if len(indices) > n_samples:
                indices = np.random.choice(indices, n_samples, replace=False)
            
            idx_in_loader = [idx % test_loader.batch_size for idx in indices]
            
            for i, idx in enumerate(idx_in_loader):
                # Use the first batch for simplicity
                batch = next(iter(test_loader))
                batch_img = batch[0][idx]
                
                # Save image for Grad-CAM
                img_file = f'analysis_results/temp_{cat_name}_{i}.png'
                plt.figure(figsize=(3, 3))
                plt.imshow(np.transpose(batch_img.numpy(), (1, 2, 0)))
                plt.axis('off')
                plt.savefig(img_file, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                img_paths.append((cat_name, img_file))
    
    # Apply Grad-CAM to selected images
    for cat_name, img_file in img_paths:
        fig, _, pred_class, pred_prob = visualize_gradcam(model, img_file, device, threshold)
        fig.suptitle(f'Category: {cat_name} | Pred: {"PNEUMONIA" if pred_class == 1 else "NORMAL"} ({pred_prob:.2f})')
        fig.savefig(f'analysis_results/gradcam_{os.path.basename(img_file)}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Analysis completed. Results saved to 'analysis_results/' directory.")

if __name__ == "__main__":
    analyze_model_predictions(threshold=0.6)