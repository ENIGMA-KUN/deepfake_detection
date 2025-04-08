import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet-b4', num_classes=2):
        super(EfficientNetModel, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        
        # Get the number of features from the last layer
        num_features = self.efficientnet._fc.in_features
        
        # Replace the final classification layer
        self.efficientnet._fc = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.efficientnet._fc(x)
        return x

def create_model(model_name='efficientnet-b4', num_classes=2, pretrained=True):
    model = EfficientNetModel(model_name=model_name, num_classes=num_classes)
    return model