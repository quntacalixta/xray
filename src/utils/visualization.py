import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_metrics(train_losses, val_losses, val_accuracies, save_path=None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        val_accuracies (list): Validation accuracies per epoch
        save_path (str, optional): Path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_predictions(model, dataloader, device, num_images=8):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing test images
        device: Device to run inference on ('cuda' or 'cpu')
        num_images (int): Number of images to display
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy for visualization
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Limit to the specified number of images
    n = min(num_images, len(images))
    
    # Create a grid to display images
    fig, axes = plt.subplots(2, n//2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Class names for display
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Display images and predictions
    for i in range(n):
        # Unnormalize image
        img = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display image and prediction info
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}\nProb: {probs[i][preds[i]]:.2f}")
        axes[i].axis('off')
        
        # Color border based on correctness
        correct = labels[i] == preds[i]
        border_color = 'green' if correct else 'red'
        for spine in axes[i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list, optional): Names of classes
        normalize (bool): Whether to normalize the confusion matrix
        title (str, optional): Title for the plot
        figsize (tuple): Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set up figure
    plt.figure(figsize=figsize)
    
    # Use seaborn for nicer styling
    sns.set(font_scale=1.2)
    
    # Create heatmap
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd', 
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        cbar=False
    )
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()