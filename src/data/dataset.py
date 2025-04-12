import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class ChestXRayImageDataset(Dataset):
    """
    Dataset class for loading chest X-ray images with additional features:
    - Supports augmentation
    - Supports caching for faster loading
    - Provides metadata about images
    """
    
    def __init__(self, 
                 image_dir, 
                 transform=None, 
                 mode='train',
                 use_cache=False,
                 seed=42):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Root directory containing images
            transform (callable, optional): Transform to apply to images
            mode (str): 'train', 'val', or 'test'
            use_cache (bool): Whether to cache images in memory
            seed (int): Random seed for reproducibility
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.use_cache = use_cache
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Get image paths and labels
        self.images, self.labels, self.metadata = self._load_dataset()
        
        # Cache for faster loading
        self.cache = {} if use_cache else None
        
    def _load_dataset(self):
        """Load dataset with paths, labels, and metadata"""
        images = []
        labels = []
        metadata = []
        
        mode_dir = os.path.join(self.image_dir, self.mode)
        
        # Class mapping
        class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        for class_name in sorted(os.listdir(mode_dir)):
            class_dir = os.path.join(mode_dir, class_name)
            
            # Skip if not a directory
            if not os.path.isdir(class_dir):
                continue
                
            # Skip if not in our class mapping
            if class_name not in class_to_idx:
                continue
                
            class_idx = class_to_idx[class_name]
            
            # Get all images in this class directory
            for img_name in sorted(os.listdir(class_dir)):
                # Check if it's an image file
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                
                # Extract metadata from filename if available
                # For example: patient_id from "patient123_date_view.jpg"
                patient_id = img_name.split('_')[0] if '_' in img_name else None
                
                # Store image path, label, and metadata
                images.append(img_path)
                labels.append(class_idx)
                metadata.append({
                    'filename': img_name,
                    'class': class_name,
                    'patient_id': patient_id
                })
        
        return images, labels, metadata
    
    def __len__(self):
        """Return the total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image by index"""
        # Check cache first if enabled
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            # Load image
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Store in cache if enabled
            if self.use_cache:
                self.cache[idx] = image
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        
        return image, label
    
    def get_metadata(self, idx):
        """Get metadata for a specific image"""
        return self.metadata[idx]
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        class_counts = np.bincount(self.labels)
        return {
            'NORMAL': int(class_counts[0]),
            'PNEUMONIA': int(class_counts[1]),
            'total': len(self.labels)
        }
    
    def get_sample_weights(self):
        """
        Calculate sample weights for weighted random sampling
        to balance classes during training
        """
        class_counts = np.bincount(self.labels)
        class_weights = 1. / class_counts
        weights = class_weights[self.labels]
        return weights

# Example usage:
# dataset = ChestXRayImageDataset(
#     image_dir='data/raw/chest_xray',
#     transform=transforms.ToTensor(),
#     mode='train'
# )
# print(f"Dataset size: {len(dataset)}")
# print(f"Class distribution: {dataset.get_class_distribution()}")
# img, label = dataset[0]
# print(f"Image shape: {img.shape}, Label: {label}")
# print(f"Metadata: {dataset.get_metadata(0)}")