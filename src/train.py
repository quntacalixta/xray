import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm
import yaml
import os
from src.data.preprocessing import create_data_loaders
from src.models.model import ChestXRayClassifier

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

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
    print(f"Using device: {device}")
    
    # Data loaders avec weighted sampling
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='data/raw/chest_xray',
        batch_size=config['training']['batch_size'],
        use_weighted_sampling=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Model
    model = ChestXRayClassifier(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Focal Loss au lieu de Cross Entropy pour gérer le déséquilibre
    criterion = FocalLoss(alpha=0.25, gamma=2)
    
    # Optimizer avec learning rate augmenté
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,  # Augmentation du LR initial
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,  # Augmentation du LR initial
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    # Learning rate scheduler plus agressif
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
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
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Logging
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        
        # Save latest model
        torch.save(checkpoint, 'checkpoints/latest_model.pth')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(checkpoint, 'best_model.pth')
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    wandb.finish()
    
    return model, train_losses, val_losses, val_accuracies

if __name__ == '__main__':
    train_model()