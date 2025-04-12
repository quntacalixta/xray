import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ChestXRayClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ChestXRayClassifier, self).__init__()
        
        # Modern torchvision weights handling
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the last fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )  # Parenthèse fermante ajoutée correctement
    
    def forward(self, x):
        return self.backbone(x)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cuda'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(num_classes=checkpoint['model_state_dict']['backbone.fc.3.weight'].shape[0])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model