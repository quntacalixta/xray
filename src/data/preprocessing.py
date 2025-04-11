import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    """Custom Dataset for Chest X-Ray Images"""
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
            mode (string): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform or self._get_default_transform()
        
        # Collect image paths and labels
        self.images, self.labels = self._load_data()
    
    def _get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_data(self):
        """Load image paths and labels"""
        images = []
        labels = []
        
        # Define class mapping
        class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # Scan through image directories
        for class_name, label in class_map.items():
            class_dir = os.path.join(self.data_dir, self.mode, class_name)
            
            # Add all images from this directory
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    images.append(img_path)
                    labels.append(label)
        
        return images, labels
    
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get a single image and its label"""
        img_path = self.images[idx]
        image = Image.open(img_path)
        
        # Apply transformations
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label

def create_data_loaders(data_dir, batch_size=32):
    """Create data loaders for train, validation, and test sets"""
    # Define transforms for different modes
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir, 
        transform=train_transform, 
        mode='train'
    )
    val_dataset = ChestXRayDataset(
        data_dir, 
        transform=val_test_transform, 
        mode='val'
    )
    test_dataset = ChestXRayDataset(
        data_dir, 
        transform=val_test_transform, 
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader