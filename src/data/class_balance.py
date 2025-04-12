import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

from src.data.dataset import ChestXRayImageDataset

def analyze_class_distribution(data_dir='data/raw/chest_xray'):
    """
    Analyze class distribution across train, validation, and test sets
    
    Args:
        data_dir (str): Path to dataset directory
    
    Returns:
        pd.DataFrame: DataFrame with class distribution statistics
    """
    distribution = {}
    
    # Analyze each split
    for mode in ['train', 'val', 'test']:
        dataset = ChestXRayImageDataset(
            image_dir=data_dir,
            mode=mode
        )
        distribution[mode] = dataset.get_class_distribution()
    
    # Create DataFrame for easy visualization
    df = pd.DataFrame(distribution).T
    
    # Calculate class ratios
    df['PNEUMONIA:NORMAL'] = df['PNEUMONIA'] / df['NORMAL']
    
    return df

def plot_class_distribution(df):
    """
    Plot class distribution 
    
    Args:
        df (pd.DataFrame): DataFrame with class distribution
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=(15, 6))
    
    # Bar chart for counts
    plt.subplot(1, 2, 1)
    df[['NORMAL', 'PNEUMONIA']].plot(kind='bar', ax=plt.gca())
    plt.title('Class Distribution Across Splits')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=0)
    
    # Line chart for class ratios
    plt.subplot(1, 2, 2)
    df['PNEUMONIA:NORMAL'].plot(kind='bar', color='purple', ax=plt.gca())
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Balanced (1:1)')
    plt.title('Class Imbalance Ratio (PNEUMONIA:NORMAL)')
    plt.ylabel('Ratio')
    plt.xticks(rotation=0)
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()

def create_balanced_sampler(dataset):
    """
    Create a weighted sampler to balance classes during training
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        WeightedRandomSampler: Sampler that balances classes
    """
    # Get sample weights (inverse frequency)
    weights = dataset.get_sample_weights()
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

def analyze_patient_distribution(data_dir='data/raw/chest_xray', mode='train'):
    """
    Analyze patient distribution to detect data leakage
    
    Args:
        data_dir (str): Path to dataset directory
        mode (str): 'train', 'val', or 'test'
        
    Returns:
        dict: Patient distribution statistics
    """
    dataset = ChestXRayImageDataset(
        image_dir=data_dir,
        mode=mode
    )
    
    # Extract patient IDs from metadata
    patient_ids = [meta['patient_id'] for meta in dataset.metadata]
    labels = dataset.labels
    
    # Count occurrences of each patient
    patient_counts = {}
    patient_labels = {}
    
    for i, patient_id in enumerate(patient_ids):
        if patient_id is None:
            continue
            
        if patient_id not in patient_counts:
            patient_counts[patient_id] = 0
            patient_labels[patient_id] = []
            
        patient_counts[patient_id] += 1
        patient_labels[patient_id].append(labels[i])
    
    # Find patients with multiple images
    multi_image_patients = {k: v for k, v in patient_counts.items() if v > 1}
    
    # Find patients with mixed labels (potential errors)
    mixed_label_patients = {}
    
    for patient_id, label_list in patient_labels.items():
        if len(set(label_list)) > 1:
            mixed_label_patients[patient_id] = label_list
    
    return {
        'total_patients': len(patient_counts),
        'multi_image_patients': len(multi_image_patients),
        'multi_image_patient_details': multi_image_patients,
        'mixed_label_patients': len(mixed_label_patients),
        'mixed_label_patient_details': mixed_label_patients
    }

def main():
    """Main function to analyze dataset class balance"""
    data_dir = 'data/raw/chest_xray'
    
    print("Analyzing class distribution...")
    df = analyze_class_distribution(data_dir)
    print("\nClass Distribution:")
    print(df)
    
    # Plot distribution
    fig = plot_class_distribution(df)
    plt.savefig('class_distribution.png')
    print("\nClass distribution plot saved to class_distribution.png")
    
    # Check for data leakage
    print("\nAnalyzing patient distribution for data leakage...")
    for mode in ['train', 'val', 'test']:
        stats = analyze_patient_distribution(data_dir, mode)
        print(f"\n{mode.upper()} set:")
        print(f"  Total patients: {stats['total_patients']}")
        print(f"  Patients with multiple images: {stats['multi_image_patients']}")
        
        if stats['mixed_label_patients'] > 0:
            print(f"  WARNING: {stats['mixed_label_patients']} patients have inconsistent labels!")
    
    # Print recommendation
    if df.loc['train', 'PNEUMONIA:NORMAL'] > 1.5 or df.loc['train', 'PNEUMONIA:NORMAL'] < 0.67:
        print("\nRECOMMENDATION: The training set has significant class imbalance.")
        print("Consider using a weighted sampler during training:")
        print("  sampler = create_balanced_sampler(train_dataset)")
        print("  train_loader = DataLoader(train_dataset, sampler=sampler, ...)")
    else:
        print("\nClass balance is acceptable.")

if __name__ == "__main__":
    main()