import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

class ImagePreprocessor:
    """
    Handles image preprocessing for deepfake detection
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize face detector
        self.face_detector = MTCNN(
            image_size=299,
            margin=40,
            device=self.device,
            keep_all=True,
            select_largest=False
        )
        
        # Setup image transformation for Xception (299x299)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Also create a 224x224 transformation for other models like CLIP
        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_image(self, image_path=None, image_data=None):
        """
        Load image from path or data
        """
        try:
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image_data is not None:
                # For uploaded files in Streamlit
                image = np.array(Image.open(image_data))
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError("Either image_path or image_data must be provided")
                
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            # Return a blank image as fallback
            return np.zeros((299, 299, 3), dtype=np.uint8)
        
    def extract_faces(self, image):
        """
        Extract faces from the input image using MTCNN
        
        Returns:
            list of tuples (face_tensor, box)
        """
        try:
            # Convert to PIL Image for MTCNN
            pil_image = Image.fromarray(image)
            
            # Detect faces
            boxes, _ = self.face_detector.detect(pil_image)
            
            faces = []
            if boxes is not None:
                for box in boxes:
                    box = box.astype(int)
                    # Extract face with a margin
                    x1, y1, x2, y2 = box
                    
                    # Ensure coordinates are within image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:  # Valid face region
                        face = Image.fromarray(image[y1:y2, x1:x2])
                        
                        # Transform the face for the model (both 299x299 and 224x224)
                        face_tensor_299 = self.transform(face).unsqueeze(0)
                        face_tensor_224 = self.transform_224(face).unsqueeze(0)
                        
                        # For compatibility with different models, we'll return both sizes
                        # The detector will use the appropriate one
                        faces.append((face_tensor_299, (x1, y1, x2, y2)))
            
            return faces
            
        except Exception as e:
            print(f"Error extracting faces: {e}")
            return []
    
    def mark_faces(self, image, faces_with_predictions):
        """
        Draw boxes and labels on the original image
        
        Args:
            image: Original image
            faces_with_predictions: List of (box, is_fake, confidence)
            
        Returns:
            Marked image
        """
        try:
            result = image.copy()
            
            for (x1, y1, x2, y2), is_fake, confidence in faces_with_predictions:
                # Set color based on prediction (red for fake, green for real)
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                
                # Draw rectangle around face
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"Fake: {confidence:.2f}" if is_fake else f"Real: {confidence:.2f}"
                
                # Position the label
                y_pos = max(y1 - 10, 20)  # Ensure label is visible
                cv2.putText(
                    result, 
                    label, 
                    (x1, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
                
            return result
        except Exception as e:
            print(f"Error marking faces: {e}")
            return image