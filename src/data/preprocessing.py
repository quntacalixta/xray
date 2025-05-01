import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform or self._get_default_transform()
        self.images, self.labels = self._load_data()
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    
    def _load_data(self):
        images, labels = [], []
        class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        for class_name, label in class_map.items():
            class_dir = os.path.join(self.data_dir, self.mode, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(class_dir, img_name))
                    labels.append(label)
        
        return images, labels
    
    def get_sample_weights(self):
        class_counts = np.bincount(self.labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        return class_weights[self.labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        return self.transform(image), self.labels[idx]

def create_data_loaders(data_dir, batch_size=32, use_weighted_sampling=True):
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    # Datasets
    train_dataset = ChestXRayDataset(data_dir, transform=train_transform, mode='train')
    val_dataset = ChestXRayDataset(data_dir, transform=val_test_transform, mode='val')
    test_dataset = ChestXRayDataset(data_dir, transform=val_test_transform, mode='test')
    
    # Weighted sampling
    sampler = None
    if use_weighted_sampling:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images, Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader

def compute_dataset_stats(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    dataset = ChestXRayDataset(data_dir, transform=transform, mode='train')
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    
    mean = 0.
    std = 0.
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()