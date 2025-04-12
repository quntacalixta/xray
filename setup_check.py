#!/usr/bin/env python3
"""
Script to verify project setup and diagnose common issues
"""

import os
import sys
import importlib
import subprocess
import platform
from pathlib import Path
import yaml
import torch

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    current_version = sys.version_info
    required_version = (3, 8)
    
    if current_version < required_version:
        print(f"❌ Python version {'.'.join(map(str, current_version[:3]))} is too old.")
        print(f"   Please use Python {'.'.join(map(str, required_version))} or newer.")
        return False
    else:
        print(f"✅ Python version {'.'.join(map(str, current_version[:3]))} is compatible.")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 
        'scikit-learn', 'Pillow', 'matplotlib', 
        'seaborn', 'wandb', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"✅ CUDA is available: Device {i}: {device_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠️ CUDA is not available. Training will use CPU only.")
        print("   This is fine for testing but will be slow for training.")
        return True  # Not a critical failure

def check_directory_structure():
    """Check if project directory structure is correct"""
    print("\nChecking directory structure...")
    required_dirs = [
        'configs', 'data', 'notebooks', 'src',
        'src/data', 'src/models', 'src/utils'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ {directory}/ exists")
        else:
            all_exist = False
            print(f"❌ {directory}/ is missing")
            
    # Check if data directories exist
    data_dirs = ['data/raw', 'data/processed', 'data/splits']
    for directory in data_dirs:
        if not os.path.isdir(directory):
            print(f"⚠️ {directory}/ is missing (will be created when needed)")
    
    return all_exist

def check_dataset():
    """Check if dataset is present in the expected location"""
    print("\nChecking dataset...")
    data_path = Path('data/raw/chest_xray')
    
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        print("   You need to download the dataset from Kaggle and extract it to data/raw/")
        return False
    
    # Check for train/val/test splits
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    missing_structure = False
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ '{split}' split missing")
            missing_structure = True
            continue
            
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"❌ '{class_name}' class missing in '{split}' split")
                missing_structure = True
                continue
                
            images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
            if not images:
                print(f"❌ No images found in {split}/{class_name}/")
                missing_structure = True
            else:
                print(f"✅ Found {len(images)} images in {split}/{class_name}/")
    
    if missing_structure:
        print("\nThe dataset should be organize