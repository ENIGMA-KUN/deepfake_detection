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

class SimpleXception(nn.Module):
    """
    Simplified Xception model for compatibility with DF40 models
    """
    def __init__(self):
        super(SimpleXception, self).__init__()
        self.model = None  # Will be loaded directly from file
        
    def forward(self, x):
        return self.model(x)

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
        self.model = None
        self.model_type = "xception"  # Default model type
        
        # Download weights if they don't exist
        if weights_path is None:
            weights_path = self.download_weights()
        
        # Try different methods to load the model
        self.load_model(weights_path)
        
    def load_model(self, weights_path):
        """
        Try different methods to load the model
        """
        if not os.path.exists(weights_path):
            print(f"Warning: Model weights not found at {weights_path}")
            return

        # Methods to try for loading the model
        methods = [
            self._load_standard_state_dict,
            self._load_direct_model,
            self._load_df40_format,
            self._load_direct_state_dict,
            self._load_jit_traced
        ]

        for method in methods:
            try:
                if method(weights_path):
                    print(f"Successfully loaded model with method: {method.__name__}")
                    return
            except Exception as e:
                print(f"Method {method.__name__} failed: {str(e)[:100]}...")

        print("All methods failed to load the model. Please check the model format.")
        
    def _load_standard_state_dict(self, weights_path):
        """Try loading as a standard state dict for XceptionNet"""
        self.model = XceptionNet(num_classes=2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "xception"
        return True
        
    def _load_direct_model(self, weights_path):
        """Try loading as a direct model"""
        self.model = torch.load(weights_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "direct"
        return True
        
    def _load_df40_format(self, weights_path):
        """Try loading in DF40 format"""
        # Load model directly
        model_data = torch.load(weights_path, map_location=self.device)
        
        # Check if it's a state dict with specific keys
        if isinstance(model_data, dict):
            if 'state_dict' in model_data:
                # Initialize the model first
                self.model = XceptionNet(num_classes=2)
                self.model.load_state_dict(model_data['state_dict'])
            else:
                # Try loading directly as a state dict
                self.model = XceptionNet(num_classes=2)
                self.model.load_state_dict(model_data)
        else:
            # Assume it's a full model
            self.model = model_data
            
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "df40"
        return True
        
    def _load_direct_state_dict(self, weights_path):
        """Try simplest form - direct load of state dict"""
        # Create a simple container model
        self.model = SimpleXception()
        # Load the model directly
        self.model.model = torch.load(weights_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "simple"
        return True
        
    def _load_jit_traced(self, weights_path):
        """Try loading as a TorchScript model"""
        self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "jit"
        return True
        
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
            if self.model is None:
                print("No model loaded. Unable to make predictions.")
                image = self.preprocessor.load_image(image_path, image_data)
                return [], image
                
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
                    
                    # Get model prediction based on model type
                    try:
                        outputs = self.model(face_tensor)
                        
                        # Handle different output formats
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
                    except Exception as e:
                        print(f"Error during prediction for face: {e}")
                        # Default to "real" with low confidence on error
                        faces_with_predictions.append((box, False, 0.5))
                
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