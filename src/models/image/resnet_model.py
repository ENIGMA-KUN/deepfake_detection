import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=2, pretrained=True):
    """
    Create a ResNet50 model for deepfake detection
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: ResNet model
    """
    # Load pre-trained ResNet50
    if pretrained:
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        model = models.resnet50(weights=None)
    
    # Modify the final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Add a dropout layer for regularization
    model.dropout = nn.Dropout(0.5)
    
    # Override forward method to include dropout
    original_forward = model.forward
    
    def new_forward(x):
        x = original_forward(x)
        return model.dropout(x)
    
    model.forward = new_forward
    
    return model 