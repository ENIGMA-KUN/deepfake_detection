import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class XceptionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        
        # Load pre-trained Xception model
        self.xception = models.xception(pretrained=True)
        
        # Modify the final classification layer
        num_features = self.xception.fc.in_features
        self.xception.fc = nn.Linear(num_features, num_classes)
        
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.xception.features(x)
        x = self.xception.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.xception.fc(x)
        return x

def create_model(num_classes=2, pretrained=True):
    model = XceptionNet(num_classes=num_classes)
    return model