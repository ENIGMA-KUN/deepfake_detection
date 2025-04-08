import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# ResNet model for deepfake detection
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Modify the final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Forward pass through resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        
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

def train_model(model_type='resnet', batch_size=8, num_epochs=3, learning_rate=0.0001):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    combined_path = os.path.join(config['paths']['processed_images'], 'combined')
    train_path = os.path.join(combined_path, 'train')
    val_path = os.path.join(combined_path, 'val')
    
    # Data transforms
    input_size = 224  # ResNet input size
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
    model = DeepfakeDetector(num_classes=2)
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
    
    # Create a simple inference model
    create_inference_script(model_type)

def create_inference_script(model_type):
    """Create a simple inference script for the model"""
    inference_code = f"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=None)
        
        # Modify the final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Forward pass through resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        
        return x

def predict_image(image_path, model_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, 1)
        
        result = {{
            'prediction': 'fake' if prediction.item() == 1 else 'real',
            'confidence': probs[0][prediction.item()].item(),
            'fake_probability': probs[0][1].item(),
            'real_probability': probs[0][0].item()
        }}
    
    # Display result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {{result['prediction'].upper()}}\\nConfidence: {{result['confidence']:.4f}}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    labels = ['Real', 'Fake']
    probs = [result['real_probability'], result['fake_probability']]
    plt.bar(labels, probs, color=['green', 'red'])
    plt.ylim(0, 1)
    plt.title('Probabilities')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with deepfake detection model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/image/{model_type}/final_model.pth', help='Path to model weights')
    
    args = parser.parse_args()
    
    # Predict
    result = predict_image(args.image, args.model)
    print(f"Prediction: {{result['prediction']}}")
    print(f"Confidence: {{result['confidence']:.4f}}")
    print(f"Real probability: {{result['real_probability']:.4f}}")
    print(f"Fake probability: {{result['fake_probability']:.4f}}")
    """
    
    # Save inference script
    with open('predict.py', 'w') as f:
        f.write(inference_code)
    
    print("Created inference script: predict.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model', type=str, default='resnet', help='Model type')
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