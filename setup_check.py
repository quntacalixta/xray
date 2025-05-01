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

    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'wandb': 'wandb',
        'yaml': 'pyyaml'
    }

    missing_packages = []
    for import_name, pip_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {pip_name} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"❌ {pip_name} is missing")

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
        print("\nThe dataset should be organized as follows:")
        print("data/raw/chest_xray/")
        print("├── train/")
        print("│   ├── NORMAL/")
        print("│   └── PNEUMONIA/")
        print("├── val/")
        print("│   ├── NORMAL/")
        print("│   └── PNEUMONIA/")
        print("└── test/")
        print("    ├── NORMAL/")
        print("    └── PNEUMONIA/")
        return False
    
    return True

def check_config():
    """Check if config file exists and is valid"""
    print("\nChecking configuration...")
    config_path = Path('configs/config.yaml')
    
    if not config_path.exists():
        print(f"❌ Configuration file not found at {config_path}")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Check required sections
        required_sections = ['model', 'training', 'preprocessing', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing '{section}' section in config")
                return False
            print(f"✅ Found '{section}' configuration")
        
        return True
    except Exception as e:
        print(f"❌ Error parsing config file: {e}")
        return False

def check_git():
    """Check git repository status"""
    print("\nChecking git status...")
    if not os.path.isdir('.git'):
        print("⚠️ Not a git repository")
        return True  # Not critical
    
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                               capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print("⚠️ You have uncommitted changes")
        else:
            print("✅ Working directory is clean")
            
        # Get current branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                               capture_output=True, text=True, check=True)
        branch = result.stdout.strip()
        print(f"✅ Current branch: {branch}")
        
        return True
    except subprocess.SubprocessError:
        print("⚠️ Failed to run git commands")
        return True  # Not critical
    except FileNotFoundError:
        print("⚠️ Git not found in PATH")
        return True  # Not critical

def check_wandb():
    """Check Weights & Biases setup"""
    print("\nChecking Weights & Biases (wandb) setup...")
    try:
        import wandb
        
        # Check if logged in
        if wandb.api.api_key:
            print("✅ WandB API key found")
        else:
            print("⚠️ Not logged in to WandB")
            print("   You can login with: wandb login")
        
        return True
    except ImportError:
        print("❌ WandB not installed")
        print("   Install with: pip install wandb")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA CLASSIFICATION PROJECT SETUP CHECK")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_cuda(),
        check_directory_structure(),
        check_dataset(),
        check_config(),
        check_git(),
        check_wandb()
    ]
    
    print("\n" + "=" * 60)
    if all(checks[:6]):  # Only the first 6 checks are critical
        print("✅ All critical checks passed! You're ready to go.")
        print("\nSuggested next steps:")
        print("1. Run the dataset analysis: python analyze_dataset.py")
        print("2. Start training: python -m src.train")
        print("3. Evaluate the model: python -m src.evaluate")
    else:
        print("❌ Some critical checks failed. Please fix the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()