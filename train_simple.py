import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# XceptionNet model
class XceptionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        
        # Load pre-trained Xception model
        self.xception = torch.hub.load('pytorch/vision:v0.10.0', 'xception', pretrained=True)
        
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

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with images organized in real/fake folders
            transform (callable, optional): Transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Get all images
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(real_dir, img_name), 0))
        
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(fake_dir, img_name), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model_type='xception', batch_size=32, num_epochs=20, learning_rate=0.0001):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    combined_path = os.path.join(config['paths']['processed_images'], 'combined')
    train_path = os.path.join(combined_path, 'train')
    val_path = os.path.join(combined_path, 'val')
    
    # Data transforms
    input_size = 299  # Xception input size
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_path, transform=data_transforms['train'])
    val_dataset = DeepfakeDataset(val_path, transform=data_transforms['val'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = XceptionNet(num_classes=2)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create model directory
    model_dir = os.path.join('models', 'image', model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(val_dataset)
        val_acc = correct / total
        
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(model_dir, 'final_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model', type=str, default='xception', help='Model type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_type=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )