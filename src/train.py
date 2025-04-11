import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import yaml
from src.data.preprocessing import create_data_loaders
from src.models.model import ChestXRayClassifier

def validate_config(config):
    """Ensure all config values have correct types"""
    # Training parameters
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    
    # Model parameters
    config['model']['num_classes'] = int(config['model']['num_classes'])
    config['model']['pretrained'] = bool(config['model']['pretrained'])
    
    # Preprocessing
    config['preprocessing']['image_size'] = int(config['preprocessing']['image_size'])
    config['preprocessing']['mean'] = [float(x) for x in config['preprocessing']['mean']]
    config['preprocessing']['std'] = [float(x) for x in config['preprocessing']['std']]
    
    return config

def train_model(config_path='configs/config.yaml'):
    # Load and validate config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = validate_config(config)

    # Initialize WandB
    wandb.init(
        project=config['logging']['project_name'],
        name=config['logging']['experiment_name'],
        config=config
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='data/raw/chest_xray',
        batch_size=config['training']['batch_size']
    )
    
    # Model
    model = ChestXRayClassifier(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        # Training
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    wandb.finish()

if __name__ == '__main__':
    train_model()