import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import io
import sys
import traceback

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Define the EfficientNet model architecture
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

# Define XceptionNet model architecture
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

class DeepfakeDetector:
    def __init__(self, model_type=config.DEFAULT_MODEL):
        """
        Initialize the deepfake detector
        
        Args:
            model_type (str): Model type ('xception' or 'efficientnet')
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize face detector
        self.face_detector = MTCNN(
            keep_all=True, 
            device=self.device,
            thresholds=[0.6, 0.7, 0.7],  # Adjust these thresholds for better face detection
            min_face_size=60
        )
        
        # Create model based on model_type
        if model_type == 'efficientnet':
            self.model = EfficientNetModel(
                model_name=config.MODEL_CONFIG['efficientnet']['name'],
                num_classes=2
            )
            self.input_size = config.MODEL_CONFIG['efficientnet']['input_size']
            model_path = config.EFFICIENTNET_PATH
        elif model_type == 'xception':
            self.model = XceptionNet(num_classes=2)
            self.input_size = config.MODEL_CONFIG['xception']['input_size']
            model_path = config.XCEPTION_PATH
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model if checkpoint exists
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                
                # If no model is available, switch to pre-trained only
                if model_type == 'efficientnet':
                    self.model = EfficientNetModel(
                        model_name=config.MODEL_CONFIG['efficientnet']['name'],
                        num_classes=2
                    )
                else:
                    self.model = XceptionNet(num_classes=2)
                print("Using pre-trained model without deepfake fine-tuning")
        else:
            print(f"No model checkpoint found at {model_path}. Using pre-trained model.")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create image transform
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_faces(self, image):
        """
        Extract faces from the input image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            list of PIL Images (faces), list of bounding boxes
        """
        try:
            # If input is PIL image, convert to numpy array
            if isinstance(image, Image.Image):
                img_np = np.array(image)
            else:
                img_np = image
                
            # Convert BGR to RGB if needed
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_np
            else:
                img_rgb = img_np
            
            # Detect faces
            boxes, _ = self.face_detector.detect(img_rgb)
            
            # If no faces detected, use the whole image
            if boxes is None:
                if isinstance(image, Image.Image):
                    return [image], [[0, 0, image.width, image.height]]
                else:
                    h, w = img_np.shape[:2]
                    return [Image.fromarray(img_rgb)], [[0, 0, w, h]]
            
            # Extract face regions
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_img = img_rgb[y1:y2, x1:x2]
                faces.append(Image.fromarray(face_img))
            
            return faces, boxes
        
        except Exception as e:
            print(f"Error in face extraction: {str(e)}")
            traceback.print_exc()
            
            # Return whole image as fallback
            if isinstance(image, Image.Image):
                return [image], [[0, 0, image.width, image.height]]
            else:
                h, w = img_np.shape[:2]
                return [Image.fromarray(img_rgb)], [[0, 0, w, h]]
    
    def predict(self, image, return_visualization=False):
        """
        Predict if an image contains deepfake faces
        
        Args:
            image: PIL Image or path to image
            return_visualization: Whether to return visualization
            
        Returns:
            dict with prediction results
        """
        try:
            # Load image if it's a path
            if isinstance(image, str):
                if not os.path.exists(image):
                    return {"error": f"Image file not found: {image}"}
                image = Image.open(image).convert("RGB")
            
            # Extract faces
            faces, boxes = self.extract_faces(image)
            
            if not faces:
                return {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "fake_probability": 0.0,
                    "real_probability": 1.0,
                    "error": "No faces detected"
                }
            
            # Process each face
            results = []
            for face, box in zip(faces, boxes):
                # Preprocess
                face_tensor = self.transform(face).unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probs, 1)
                    
                    result = {
                        "prediction": "fake" if prediction.item() == 1 else "real",
                        "confidence": confidence.item(),
                        "fake_probability": probs[0][1].item(),
                        "real_probability": probs[0][0].item(),
                        "box": box
                    }
                    results.append(result)
            
            # Combine results (use the face with highest fake probability for overall result)
            max_fake_prob = max(r["fake_probability"] for r in results)
            max_fake_idx = next(i for i, r in enumerate(results) if r["fake_probability"] == max_fake_prob)
            
            overall_result = results[max_fake_idx].copy()
            overall_result["all_faces"] = results
            
            # Generate visualization if requested
            if return_visualization:
                fig, buffer = self.create_visualization(image, results)
                overall_result["visualization"] = buffer
                overall_result["fig"] = fig
            
            return overall_result
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return {
                "error": str(e),
                "prediction": "unknown",
                "confidence": 0.0
            }
    
    def create_visualization(self, image, results):
        """
        Create visualization of detection results
        
        Args:
            image: PIL Image
            results: List of detection results
        
        Returns:
            matplotlib figure and image buffer
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Display image
        img_np = np.array(image)
        ax.imshow(img_np)
        
        # Draw bounding boxes and labels
        for result in results:
            box = result["box"]
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            # Set color based on prediction (green for real, red for fake)
            color = 'green' if prediction == 'real' else 'red'
            
            # Draw bounding box
            rect = plt.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1], 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{prediction.upper()} ({confidence:.2f})"
            ax.text(
                box[0], 
                box[1] - 10, 
                label, 
                color='white', 
                fontsize=12, 
                bbox=dict(facecolor=color, alpha=0.8)
            )
        
        ax.set_title("Deepfake Detection Results")
        ax.axis('off')
        
        # Save figure to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        return fig, buffer

# Helper function to download a sample pre-trained model if none exists
def download_sample_model(model_type=config.DEFAULT_MODEL):
    """
    Download a sample pre-trained model if none exists
    
    Args:
        model_type (str): Model type ('xception' or 'efficientnet')
    """
    # In a real application, you would download a model from a server
    # or use a pre-trained model from a model repository
    pass