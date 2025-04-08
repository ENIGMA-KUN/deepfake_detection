import os
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.nn import functional as F
import sys

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import models
from src.models.image.xception_model import create_model as create_xception
from src.models.image.efficientnet_model import create_model as create_efficientnet
from src.models.image.resnet_model import create_model as create_resnet

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

class DeepfakeDetector:
    def __init__(self, model_type='xception', checkpoint_path=None, device=None):
        """
        Initialize the deepfake detector
        
        Args:
            model_type (str): Model type ('xception', 'efficientnet', or 'resnet')
            checkpoint_path (str): Path to model checkpoint
            device (torch.device): Device to use
        """
        self.model_type = model_type
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Set input size based on model type
        if model_type == 'xception':
            self.input_size = config['models']['image']['xception']['input_size']
            self.model = create_xception(num_classes=2)
        elif model_type == 'efficientnet':
            self.input_size = config['models']['image']['efficientnet']['input_size']
            self.model = create_efficientnet(num_classes=2)
        elif model_type == 'resnet':
            # Use default ResNet input size of 224x224 if not specified in config
            self.input_size = config.get('models', {}).get('image', {}).get('resnet', {}).get('input_size', 224)
            self.model = create_resnet(num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            # Try to load default checkpoint
            default_path = os.path.join('models', 'image', model_type, 'best_model_acc.pth')
            if os.path.exists(default_path):
                self.load_checkpoint(default_path)
            else:
                print(f"No checkpoint found at {default_path}. Using untrained model.")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    def extract_face(self, image_path):
        """
        Extract face from image using simple method (for testing)
        
        Args:
            image_path (str): Path to image or numpy array
        
        Returns:
            PIL.Image: Extracted face
            tuple: Face bounding box (x, y, w, h)
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
            else:
                raise ValueError("Invalid image format")
        
        # For testing, just use the central region of the image as a "face"
        h, w = image_rgb.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 2
        
        # Create bounding box
        x = max(0, center_x - size)
        y = max(0, center_y - size)
        width = min(size * 2, w - x)
        height = min(size * 2, h - y)
        
        # Extract face
        face = image_rgb[y:y+height, x:x+width]
        
        # Convert to PIL image
        pil_face = Image.fromarray(face)
        
        return pil_face, (x, y, width, height)
    
    def predict(self, image_path, return_heatmap=False):
        """
        Predict if an image is real or fake
        
        Args:
            image_path (str): Path to image or numpy array
            return_heatmap (bool): Whether to return attention heatmap
            
        Returns:
            dict: Prediction results
        """
        try:
            # Extract face
            face, bbox = self.extract_face(image_path)
            
            # Preprocess
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probs, 1)
                
                result = {
                    'prediction': 'fake' if prediction.item() == 1 else 'real',
                    'confidence': confidence.item(),
                    'fake_probability': probs[0][1].item(),
                    'real_probability': probs[0][0].item(),
                    'bbox': bbox
                }
                
                # Generate simple heatmap if requested
                if return_heatmap:
                    # Create a simple gradient-based heatmap for visualization
                    heatmap_np = np.zeros((face.height, face.width))
                    
                    # Create a diagonal gradient as a placeholder
                    for i in range(face.height):
                        for j in range(face.width):
                            dist_from_center = abs(i - face.height//2) + abs(j - face.width//2)
                            heatmap_np[i, j] = 1.0 - min(1.0, dist_from_center / (face.height//2 + face.width//2))
                    
                    # Apply colormap
                    heatmap_cv = np.uint8(255 * heatmap_np)
                    heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)
                    
                    # Convert face to numpy array
                    face_np = np.array(face)
                    
                    # Overlay heatmap on face
                    overlay = cv2.addWeighted(face_np, 0.7, heatmap_cv, 0.3, 0)
                    
                    result['heatmap'] = {
                        'raw': heatmap_np,
                        'colored': heatmap_cv,
                        'overlay': overlay
                    }
                
                return result
        
        except Exception as e:
            return {
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0
            }
    
    def visualize_prediction(self, image_path, output_path=None):
        """
        Visualize prediction results
        
        Args:
            image_path (str): Path to image or numpy array
            output_path (str): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        try:
            # Read original image
            if isinstance(image_path, str):
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            else:
                original_image = image_path.copy()
                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.shape[2] == 3 else original_image
            
            # Get prediction with heatmap
            result = self.predict(image_path, return_heatmap=True)
            
            if 'error' in result:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.imshow(original_image)
                ax.set_title(f"Error: {result['error']}")
                ax.axis('off')
            else:
                # Draw bounding box on original image
                x, y, w, h = result['bbox']
                image_with_bbox = original_image.copy()
                color = (0, 255, 0) if result['prediction'] == 'real' else (255, 0, 0)
                cv2.rectangle(image_with_bbox, (x, y), (x+w, y+h), color, 2)
                
                # Add text
                text = f"{result['prediction'].upper()}: {result['confidence']:.2f}"
                cv2.putText(image_with_bbox, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Create figure
                if 'heatmap' in result:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image with bbox
                    axs[0].imshow(image_with_bbox)
                    axs[0].set_title(f"Prediction: {result['prediction'].upper()}")
                    axs[0].axis('off')
                    
                    # Heatmap
                    axs[1].imshow(result['heatmap']['raw'], cmap='jet')
                    axs[1].set_title('Attention Heatmap')
                    axs[1].axis('off')
                    
                    # Overlay
                    axs[2].imshow(result['heatmap']['overlay'])
                    axs[2].set_title('Overlay')
                    axs[2].axis('off')
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(image_with_bbox)
                    ax.set_title(f"Prediction: {result['prediction'].upper()}")
                    ax.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")
            
            return fig
        
        except Exception as e:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            return fig

def batch_evaluate(model_type, test_dir, output_dir=None):
    """
    Evaluate model on a test directory
    
    Args:
        model_type (str): Model type ('xception' or 'efficientnet')
        test_dir (str): Path to test directory
        output_dir (str): Path to save results
    """
    import time
    
    # Initialize detector
    detector = DeepfakeDetector(model_type=model_type)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join('results', 'image', model_type, time.strftime('%Y%m%d_%H%M%S'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get real and fake directories
    real_dir = os.path.join(test_dir, 'real')
    fake_dir = os.path.join(test_dir, 'fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise ValueError(f"Test directory structure is invalid. Expected 'real' and 'fake' subdirectories in {test_dir}")
    
    # Get image files
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images")
    
    # Evaluate on real images
    real_results = []
    for img_path in real_files:
        try:
            result = detector.predict(img_path)
            result['path'] = img_path
            result['true_label'] = 'real'
            real_results.append(result)
            
            if 'error' not in result:
                # Save visualization
                output_path = os.path.join(output_dir, f"real_{os.path.basename(img_path)}")
                detector.visualize_prediction(img_path, output_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Evaluate on fake images
    fake_results = []
    for img_path in fake_files:
        try:
            result = detector.predict(img_path)
            result['path'] = img_path
            result['true_label'] = 'fake'
            fake_results.append(result)
            
            if 'error' not in result:
                # Save visualization
                output_path = os.path.join(output_dir, f"fake_{os.path.basename(img_path)}")
                detector.visualize_prediction(img_path, output_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Combine results
    all_results = real_results + fake_results
    
    # Calculate metrics
    correct = sum(1 for r in all_results if 'error' not in r and r['prediction'] == r['true_label'])
    total = sum(1 for r in all_results if 'error' not in r)
    accuracy = correct / total if total > 0 else 0
    
    real_correct = sum(1 for r in real_results if 'error' not in r and r['prediction'] == 'real')
    real_total = sum(1 for r in real_results if 'error' not in r)
    real_accuracy = real_correct / real_total if real_total > 0 else 0
    
    fake_correct = sum(1 for r in fake_results if 'error' not in r and r['prediction'] == 'fake')
    fake_total = sum(1 for r in fake_results if 'error' not in r)
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
    
    # Create summary
    summary = {
        'model_type': model_type,
        'total_images': len(all_results),
        'processed_images': total,
        'failed_images': len(all_results) - total,
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'real_images': len(real_results),
        'fake_images': len(fake_results),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation completed. Results saved to {output_dir}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Real accuracy: {real_accuracy:.4f}")
    print(f"Fake accuracy: {fake_accuracy:.4f}")
    
    return summary, all_results

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Inference with deepfake detection model')
    parser.add_argument('--model', type=str, default='xception', choices=['xception', 'efficientnet', 'resnet'],
                        help='Model type (xception, efficientnet, or resnet)')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output visualization')
    parser.add_argument('--test_dir', type=str, help='Path to test directory for batch evaluation')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image prediction
        detector = DeepfakeDetector(model_type=args.model)
        result = detector.predict(args.image)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Fake probability: {result['fake_probability']:.4f}")
            print(f"Real probability: {result['real_probability']:.4f}")
        
        if args.output:
            detector.visualize_prediction(args.image, args.output)
    
    elif args.test_dir:
        # Batch evaluation
        output_dir = os.path.join('results', 'image', args.model, time.strftime('%Y%m%d_%H%M%S'))
        batch_evaluate(args.model, args.test_dir, output_dir)
    
    else:
        parser.print_help()