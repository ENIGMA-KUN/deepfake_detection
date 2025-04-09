import os
import torch
import torch.nn as nn
import numpy as np
import gdown
from PIL import Image
import pretrainedmodels
from src.preprocessing import ImagePreprocessor

class XceptionNet(nn.Module):
    """
    Xception model adapted for deepfake detection
    """
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        # Load the Xception model from pretrainedmodels library
        self.xception = pretrainedmodels.xception(pretrained='imagenet')
        
        # Modify the final classifier for our binary task
        feature_size = self.xception.last_linear.in_features
        self.xception.last_linear = nn.Linear(feature_size, num_classes)
        
    def forward(self, x):
        return self.xception(x)

class DeepfakeDetector:
    """
    Main deepfake detection class that handles the model and preprocessing
    """
    def __init__(self, weights_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize the preprocessor
        self.preprocessor = ImagePreprocessor(device=self.device)
        
        # Initialize our deepfake detection model
        self.model = XceptionNet(num_classes=2)
        
        # Download weights if they don't exist
        if weights_path is None:
            weights_path = self.download_weights()
            
        # Load model weights with improved error handling
        if os.path.exists(weights_path):
            try:
                # Try standard loading first
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"Successfully loaded model weights from {weights_path}")
            except Exception as e:
                print(f"Standard loading failed: {e}")
                try:
                    # Try loading as full model (not just state dict)
                    loaded_model = torch.load(weights_path, map_location=self.device)
                    
                    # Check if it's already a complete model or just state dict
                    if isinstance(loaded_model, dict) and 'state_dict' in loaded_model:
                        self.model.load_state_dict(loaded_model['state_dict'])
                    elif isinstance(loaded_model, dict):
                        self.model.load_state_dict(loaded_model)
                    else:
                        # It's a full model object
                        self.model = loaded_model
                    print(f"Successfully loaded model using alternative method from {weights_path}")
                except Exception as e2:
                    print(f"All loading methods failed: {e2}")
                    print("Please ensure your model file is compatible with the XceptionNet architecture")
        else:
            print(f"Warning: Model weights not found at {weights_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def download_weights(self):
        """
        Download pretrained model weights if they don't exist
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check for both potential filenames
        weights_path = 'models/xception.pth'
        alt_weights_path = 'models/xception_deepfake.pth'
        
        # Use existing file if available
        if os.path.exists(weights_path):
            print(f"Model weights found at {weights_path}")
            return weights_path
        elif os.path.exists(alt_weights_path):
            print(f"Model weights found at {alt_weights_path}")
            return alt_weights_path
            
        # If no file exists, download it
        print("Downloading model weights...")
        target_path = 'models/xception_deepfake.pth'
        url = 'https://drive.google.com/uc?id=1lPMhLMGdQRZT9jKd_CRXSiOvNCTXRbnD'
        
        try:
            gdown.download(url, target_path, quiet=False)
            print(f"Downloaded model weights to {target_path}")
            return target_path
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please ensure the model file 'xception.pth' is in the models/ directory")
            # Return the path anyway in case the file was partially downloaded
            return target_path
    
    def predict(self, image_path=None, image_data=None):
        """
        Detect deepfakes in an image
        
        Args:
            image_path: Path to the image file
            image_data: Image data (for uploaded files in Streamlit)
            
        Returns:
            faces_with_predictions: List of (box, is_fake, confidence)
            marked_image: Image with boxes and labels
        """
        try:
            with torch.no_grad():
                # Load and preprocess the image
                image = self.preprocessor.load_image(image_path, image_data)
                
                # Extract faces
                faces = self.preprocessor.extract_faces(image)
                
                if not faces:
                    print("No faces detected in the image")
                    return [], image
                
                faces_with_predictions = []
                for face_tensor, box in faces:
                    # Move tensor to device
                    face_tensor = face_tensor.to(self.device)
                    
                    # Get model prediction
                    outputs = self.model(face_tensor)
                    
                    # Handle different output formats (some models might return logits, others probabilities)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Some models return (outputs, features)
                    
                    # Convert to probabilities if needed
                    if outputs.shape[1] == 2:  # Binary classification
                        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                        # Get prediction (0: real, 1: fake)
                        is_fake = probabilities[1] > 0.5
                        confidence = probabilities[1] if is_fake else probabilities[0]
                    else:  # Single output for binary classification
                        confidence = torch.sigmoid(outputs).cpu().numpy()[0][0]
                        is_fake = confidence > 0.5
                    
                    faces_with_predictions.append((box, is_fake, confidence))
                
                # Mark faces on the image
                marked_image = self.preprocessor.mark_faces(image, faces_with_predictions)
                
                return faces_with_predictions, marked_image
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return empty predictions and original image in case of error
            if image_path:
                image = self.preprocessor.load_image(image_path)
            elif image_data is not None:
                image = self.preprocessor.load_image(image_data=image_data)
            else:
                # Create a blank image if both are None
                image = np.zeros((300, 300, 3), dtype=np.uint8)
            return [], image

    def get_prediction_explanation(self, confidence):
        """
        Generate an explanation of the prediction based on the confidence score
        """
        if confidence > 0.95:
            return "Very high confidence: The model is extremely certain about this prediction."
        elif confidence > 0.85:
            return "High confidence: The model has strong evidence for this prediction."
        elif confidence > 0.7:
            return "Moderate confidence: The model has good evidence for this prediction."
        elif confidence > 0.6:
            return "Low confidence: The model is somewhat uncertain about this prediction."
        else:
            return "Very low confidence: The model is highly uncertain about this prediction."