
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
        
        result = {
            'prediction': 'fake' if prediction.item() == 1 else 'real',
            'confidence': probs[0][prediction.item()].item(),
            'fake_probability': probs[0][1].item(),
            'real_probability': probs[0][0].item()
        }
    
    # Display result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {result['prediction'].upper()}\nConfidence: {result['confidence']:.4f}")
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
    parser.add_argument('--model', type=str, default='models/image/resnet/final_model.pth', help='Path to model weights')
    
    args = parser.parse_args()
    
    # Predict
    result = predict_image(args.image, args.model)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Real probability: {result['real_probability']:.4f}")
    print(f"Fake probability: {result['fake_probability']:.4f}")
    