import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registering hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward()
        
        # Compute CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activation maps with gradients
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average over channels
        cam = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam, target_class

def apply_colormap(cam, img):
    """Apply colormap to CAM and overlay on image"""
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    # Convert to RGB if img is grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Overlay
    cam = cv2.addWeighted(img, 0.7, cam, 0.3, 0)
    return cam

def visualize_gradcam(model, img_path, device, threshold=0.6):
    """Visualize Grad-CAM for a given image"""
    # Load image
    from torchvision import transforms
    from src.data.preprocessing import ChestXRayDataset
    
    # Get transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    # Process image
    img = Image.open(img_path)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get target layer
    target_layer = model.backbone.layer4[-1]
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, class_idx = grad_cam.generate_cam(input_tensor)
    
    # Get prediction probability
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_prob = probs[0, class_idx].item()
        pred_class = 1 if pred_prob > threshold else 0
    
    # Convert for visualization
    img_np = np.array(img.convert('RGB'))
    vis = apply_colormap(cam, img_np)
    
    # Plot
    class_names = ['NORMAL', 'PNEUMONIA']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM
    axes[1].imshow(vis)
    axes[1].set_title(f'Prediction: {class_names[pred_class]} ({pred_prob:.2f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    return fig, cam, pred_class, pred_prob