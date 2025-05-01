#!/usr/bin/env python3
"""
Script for making predictions on new chest X-ray images
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.models.model import ChestXRayClassifier

def preprocess_image(image_path, size=224):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image
        size (int): Size to resize the image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define transformations (same as used during training)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocessed = transform(image)
    
    # Add batch dimension
    preprocessed = preprocessed.unsqueeze(0)
    
    return preprocessed, image

def predict(model, image_tensor, threshold=0.5, device='cpu'):
    """
    Make a prediction on an image
    
    Args:
        model: Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        threshold (float): Classification threshold
        device (str): Device to run inference on
        
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

def visualize_prediction(image, prediction, probability, save_path=None):
    class_names = ['NORMAL', 'PNEUMONIA']
    class_name = class_names[prediction]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    
    # Set title color based on prediction
    title_color = 'green' if prediction == 0 else 'red'
    ax.set_title(f'Prediction: {class_name} ({probability:.2%})', 
                color=title_color, fontsize=16)
    
    # Add confidence text
    plt.figtext(0.5, 0.01, f'NORMAL {(1-probability):.2%} | {probability:.2%} PNEUMONIA', 
               ha='center', fontsize=12)
    
    # Properly create colorbar
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='horizontal', label='Pneumonia Probability')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Important pour éviter les fuites de mémoire
    
    return fig

def process_batch(model, image_paths, threshold=0.5, output_dir=None, device='cpu'):
    """
    Process a batch of images
    
    Args:
        model: Trained model
        image_paths (list): List of image paths
        threshold (float): Classification threshold
        output_dir (str, optional): Directory to save visualizations
        device (str): Device to run inference on
        
    Returns:
        list: Predictions for all images
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path)
        
        # Make prediction
        prediction, probability = predict(model, image_tensor, threshold, device)
        
        # Create result dictionary
        result = {
            'image_path': image_path,
            'prediction': prediction,
            'probability': probability,
            'class_name': 'NORMAL' if prediction == 0 else 'PNEUMONIA'
        }
        
        results.append(result)
        
        # Visualize if output directory is specified
        if output_dir:
            output_path = os.path.join(output_dir, f"pred_{Path(image_path).stem}.png")
            visualize_prediction(original_image, prediction, probability, output_path)
            print(f"  Saved visualization to {output_path}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Chest X-Ray Pneumonia Prediction")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--batch", type=str, help="Directory containing images for batch prediction")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--output", type=str, help="Output directory for visualizations")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for inference")
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be specified")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    try:
        print(f"Loading model from {args.model}...")
        model = ChestXRayClassifier.load_from_checkpoint(args.model, device=device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        
        print(f"Processing image: {args.image}")
        image_tensor, original_image = preprocess_image(args.image)
        prediction, probability = predict(model, image_tensor, args.threshold, device)
        
        print(f"Prediction: {'NORMAL' if prediction == 0 else 'PNEUMONIA'}")
        print(f"Probability: {probability:.2%}")
        
        if args.output:
            output_path = os.path.join(args.output, f"pred_{Path(args.image).stem}.png")
            visualize_prediction(original_image, prediction, probability, output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    # Process batch of images
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Directory not found: {args.batch}")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(args.batch).glob(f"*{ext}")))
        
        if not image_paths:
            print(f"No images found in {args.batch}")
            return
        
        print(f"Found {len(image_paths)} images")
        results = process_batch(model, image_paths, args.threshold, args.output, device)
        
        # Print summary
        normal_count = sum(1 for r in results if r['prediction'] == 0)
        pneumonia_count = sum(1 for r in results if r['prediction'] == 1)
        
        print("\nPrediction Summary:")
        print(f"  NORMAL: {normal_count} images")
        print(f"  PNEUMONIA: {pneumonia_count} images")
        
        # Save results to CSV
        if args.output:
            import csv
            csv_path = os.path.join(args.output, "batch_predictions.csv")
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'prediction', 'probability', 'class_name'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()