import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def create_data_structure():
    """Create the required directory structure"""
    # Define the paths
    base_path = Path("data")
    paths = {
        "raw": base_path / "raw" / "chest_xray",
        "processed": base_path / "processed",
        "splits": base_path / "splits"
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def analyze_dataset(data_path):
    """Analyze the dataset structure and print summary"""
    summary = {"train": {}, "val": {}, "test": {}}
    
    # Collect statistics
    for split in ["train", "val", "test"]:
        split_path = data_path / split
        if split_path.exists():
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_path = split_path / class_name
                if class_path.exists():
                    n_images = len(list(class_path.glob("*.jpeg")))
                    summary[split][class_name] = n_images
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary).fillna(0).astype(int)
    
    return df_summary

def display_sample_images(data_path):
    """Display sample images from each class"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    classes = ["NORMAL", "PNEUMONIA"]
    
    for i, class_name in enumerate(classes):
        # Get one image from train and one from test
        train_img_path = next((data_path / "train" / class_name).glob("*.jpeg"))
        test_img_path = next((data_path / "test" / class_name).glob("*.jpeg"))
        
        # Display images
        axes[i, 0].imshow(Image.open(train_img_path), cmap='gray')
        axes[i, 0].set_title(f'Train - {class_name}')
        axes[i, 1].imshow(Image.open(test_img_path), cmap='gray')
        axes[i, 1].set_title(f'Test - {class_name}')
    
    plt.tight_layout()
    return fig

def main():
    # Create directory structure
    paths = create_data_structure()
    
    # After you've downloaded and extracted the dataset to data/raw/chest_xray
    data_path = paths["raw"]
    
    # Analyze dataset
    summary_df = analyze_dataset(data_path)
    print("\nDataset Summary:")
    print(summary_df)
    
    # Display sample images
    fig = display_sample_images(data_path)
    plt.show()

if __name__ == "__main__":
    main()