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
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def load_image(self, image_path=None, image_data=None):
        """
        Load image from path or data
        """
        if image_path:
            image = cv2.imread(image_path)
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
                        # Transform the face for the model
                        face_tensor = self.transform(face).unsqueeze(0)
                        faces.append((face_tensor, (x1, y1, x2, y2)))
            
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
        result = image.copy()
        
        for (x1, y1, x2, y2), is_fake, confidence in faces_with_predictions:
            # Set color based on prediction (red for fake, green for real)
            color = (0, 0, 255) if is_fake else (0, 255, 0)
            
            # Draw rectangle around face
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"Fake: {confidence:.2f}" if is_fake else f"Real: {confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return result