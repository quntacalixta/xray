#!/usr/bin/env python3
"""
Script to test the pneumonia detection model on external images.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib.gridspec as gridspec

# Import your model - adjust the import path as needed
from src.models.model import ChestXRayClassifier

def preprocess_image(image_path, size=224):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image
        size (int): Size to resize the image to
        
    Returns:
        tuple: (preprocessed tensor, original image)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25]
        )
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocessed = transform(image)
    
    # Add batch dimension
    preprocessed = preprocessed.unsqueeze(0)
    
    return preprocessed, image

def predict_image(model, image_tensor, threshold=0.7, device='cpu'):
    """
    Make a prediction on an image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        threshold: Classification threshold
        device: Device to run inference on
        
    Returns:
        tuple: (prediction, probability)
    """
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get probability for pneumonia class
        pneumonia_prob = probabilities[0, 1].item()
        
        # Make prediction based on threshold
        prediction = int(pneumonia_prob > threshold)
    
    return prediction, pneumonia_prob

def visualize_results(images, predictions, probabilities, image_paths, threshold, output_dir=None):
    """
    Visualize prediction results
    
    Args:
        images: List of original images
        predictions: List of predictions (0/1)
        probabilities: List of prediction probabilities
        image_paths: List of image paths
        threshold: Threshold used for classification
        output_dir: Directory to save the visualization
    """
    # Calculate grid dimensions
    n_images = len(images)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(15, 5 * n_rows))
    
    # Create grid
    for i, (img, pred, prob, path) in enumerate(zip(images, predictions, probabilities, image_paths)):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Display image
        plt.imshow(img)
        
        # Set title color based on prediction
        title_color = 'green' if pred == 0 else 'red'
        filename = os.path.basename(path)
        
        plt.title(f"Prediction: {'NORMAL' if pred == 0 else 'PNEUMONIA'}\nProbability: {prob:.2f}\n{filename}", 
                 color=title_color, fontsize=12)
        
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"external_test_results_thresh{threshold}.png"))
        print(f"Visualization saved to {output_dir}/external_test_results_thresh{threshold}.png")
    
    plt.show()

def test_external_images(image_dir, model_path, threshold=0.7, output_dir=None):
    """
    Test model on a directory of external images
    
    Args:
        image_dir: Directory containing test images
        model_path: Path to the trained model
        threshold: Classification threshold
        output_dir: Directory to save results
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = ChestXRayClassifier.load_from_checkpoint(model_path, device=device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.lower().endswith(ext)])
    
    if not image_paths:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    images = []
    processed_tensors = []
    
    for path in image_paths:
        processed, original = preprocess_image(path)
        processed_tensors.append(processed)
        images.append(original)
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for tensor in processed_tensors:
        pred, prob = predict_image(model, tensor, threshold, device)
        predictions.append(pred)
        probabilities.append(prob)
    
    # Print results
    print("\nPrediction Results:")
    for i, (path, pred, prob) in enumerate(zip(image_paths, predictions, probabilities)):
        print(f"{i+1}. {os.path.basename(path)}: {'NORMAL' if pred == 0 else 'PNEUMONIA'} (Probability: {prob:.4f})")
    
    # Count results
    normal_count = predictions.count(0)
    pneumonia_count = predictions.count(1)
    
    print(f"\nSummary: {normal_count} images classified as NORMAL, {pneumonia_count} as PNEUMONIA")
    
    # Create visualization
    visualize_results(images, predictions, probabilities, image_paths, threshold, output_dir)
    
    # Save detailed results to CSV if output directory is specified
    if output_dir:
        import csv
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"external_test_results_thresh{threshold}.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Prediction', 'Probability'])
            
            for path, pred, prob in zip(image_paths, predictions, probabilities):
                writer.writerow([os.path.basename(path), 
                                'NORMAL' if pred == 0 else 'PNEUMONIA', 
                                f"{prob:.4f}"])
            
        print(f"Detailed results saved to {output_dir}/external_test_results_thresh{threshold}.csv")
    
def main():
    parser = argparse.ArgumentParser(description="Test pneumonia detection model on external images")
    parser.add_argument("--image_dir", type=str, default="data/external_test", 
                        help="Directory containing test images")
    parser.add_argument("--model", type=str, default="best_model.pth", 
                        help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Classification threshold")
    parser.add_argument("--output", type=str, default="results/external_test", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    test_external_images(args.image_dir, args.model, args.threshold, args.output)

if __name__ == "__main__":
    main()