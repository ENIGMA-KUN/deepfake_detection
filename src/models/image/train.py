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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import models
from src.models.image.xception_model import create_model as create_xception
from src.models.image.efficientnet_model import create_model as create_efficientnet
# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

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

def train_model(model_type='xception', batch_size=32, num_epochs=20, learning_rate=0.0001, device=None):
    """
    Train a deepfake detection model
    
    Args:
        model_type (str): Model type ('xception' or 'efficientnet')
        batch_size (int): Batch size
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (torch.device): Device to use
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Set paths
    combined_path = os.path.join(config['paths']['processed_images'], 'combined')
    train_path = os.path.join(combined_path, 'train')
    val_path = os.path.join(combined_path, 'val')
    
    # Check if datasets exist
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training or validation data not found. Please run data preprocessing first.")
    
    # Set transforms based on model type
    if model_type == 'xception':
        input_size = config['models']['image']['xception']['input_size']
        model_create_fn = create_xception
    elif model_type == 'efficientnet':
        input_size = config['models']['image']['efficientnet']['input_size']
        model_create_fn = create_efficientnet
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = model_create_fn(num_classes=2)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    # Create model directory
    model_dir = os.path.join('models', 'image', model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join('logs', 'image', model_type)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting training {model_type} model for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
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
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = accuracy_score(y_true_train, y_pred_train)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        y_true_val = []
        y_pred_val = []
        y_scores_val = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                
                # Save scores for ROC-AUC calculation
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_scores_val.extend(scores)
        
        val_loss = running_loss / len(val_dataset)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_precision = precision_score(y_true_val, y_pred_val, zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
        
        # Calculate AUC only if both classes are present
        if len(set(y_true_val)) > 1:
            val_auc = roc_auc_score(y_true_val, y_scores_val)
        else:
            val_auc = 0.0
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.0f}s")
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Save model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model_loss.pth'))
            patience_counter = 0
            print(f"New best model saved (loss)")
        else:
            patience_counter += 1
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model_acc.pth'))
            print(f"New best model saved (accuracy)")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(model_dir, 'final_model.pth')}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_history.png'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc
    }
    
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to {os.path.join(log_dir, 'training_history.json')}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model', type=str, default='xception', choices=['xception', 'efficientnet'],
                        help='Model type (xception or efficientnet)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_type=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )