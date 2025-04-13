#!/usr/bin/env python3
"""
Main execution script for the Chest X-Ray Pneumonia Classification project.
This script provides a unified interface to run all steps of the project.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def setup_environment():
    """Setup the project environment"""
    print_header("Setting Up Environment")
    
    # Run setup check
    print("Running setup check...")
    setup_check_result = os.system("python setup_check.py")
    
    if setup_check_result != 0:
        print("❌ Setup check failed. Please fix the issues before continuing.")
        return False
    
    # Create necessary directories
    for directory in ['data/raw', 'data/processed', 'data/splits', 'checkpoints', 'results']:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def analyze_dataset():
    """Run dataset analysis"""
    print_header("Analyzing Dataset")
    
    # Check if dataset exists
    if not Path('data/raw/chest_xray').exists():
        print("❌ Dataset not found at data/raw/chest_xray")
        return False
    
    # Run dataset analysis
    print("Running dataset analysis...")
    os.system("python analyze_dataset.py")
    
    # Run class balance analysis
    print("\nAnalyzing class balance...")
    os.system("python -m src.data.class_balance")
    
    return True

def train_model(config_path):
    """Train the model"""
    print_header("Training Model")
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"❌ Config not found at {config_path}")
        return False
    
    # Load config to display parameters
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("Training with the following configuration:")
    print(f"Model: {config['model']['name']} (pretrained: {config['model']['pretrained']})")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Optimizer: {config['training']['optimizer']}")
    
    # Start timer
    start_time = time.time()
    
    # Run training
    os.system(f"python -m src.train --config {config_path}")
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return True

def evaluate_model(thresholds=None):
    """Evaluate the model"""
    print_header("Evaluating Model")
    
    # Check if model exists
    if not Path('best_model.pth').exists():
        print("❌ Model not found at best_model.pth")
        return False
    
    # Run evaluation with multiple thresholds if specified
    if thresholds:
        for threshold in thresholds:
            print(f"\nEvaluating with threshold {threshold}...")
            os.system(f"python -m src.evaluate --threshold {threshold}")
    else:
        print("Evaluating with default threshold...")
        os.system("python -m src.evaluate")
    
    return True

def visualize_results():
    """Generate visualizations of results"""
    print_header("Visualizing Results")
    
    # Create directory for visualizations
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Run the results analysis notebook as a script
    print("Generating visualizations...")
    os.system("jupyter nbconvert --to python notebooks/03_results_analysis.ipynb")
    os.system("python notebooks/03_results_analysis.py")
    
    print("✅ Visualizations generated in results/visualizations/")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Chest X-Ray Pneumonia Classification Project")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--thresholds", type=float, nargs="+", help="Thresholds for evaluation")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Run all steps if --all is specified
    if args.all:
        args.setup = args.analyze = args.train = args.evaluate = args.visualize = True
    
    # Execute requested steps
    if args.setup and not setup_environment():
        print("❌ Setup failed. Exiting.")
        return
    
    if args.analyze and not analyze_dataset():
        print("❌ Dataset analysis failed. Exiting.")
        return
    
    if args.train and not train_model(args.config):
        print("❌ Training failed. Exiting.")
        return
    
    if args.evaluate and not evaluate_model(args.thresholds):
        print("❌ Evaluation failed. Exiting.")
        return
    
    if args.visualize and not visualize_results():
        print("❌ Visualization failed. Exiting.")
        return
    
    if any([args.setup, args.analyze, args.train, args.evaluate, args.visualize]):
        print_header("Complete")
        print("✅ All requested steps have been completed successfully!")
        print("\nTo use the trained model for prediction:")
        print("  1. Load the model: model = ChestXRayClassifier.load_from_checkpoint('best_model.pth')")
        print("  2. Preprocess your image: image = preprocess_image('your_image.jpg')")
        print("  3. Make prediction: prediction = model(image)")

if __name__ == "__main__":
    main()