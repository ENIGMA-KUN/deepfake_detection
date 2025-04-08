import os
import sys
import unittest
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import detector
from src.models.image.inference import DeepfakeDetector

class TestDeepfakeDetector(unittest.TestCase):
    def setUp(self):
        # Create a dummy detector (without loading model weights)
        self.detector = DeepfakeDetector(model_type='xception')
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.model_type, 'xception')
    
    def test_create_dummy_image(self):
        """Test creating a dummy image for detection"""
        # Create a dummy image
        img = Image.new('RGB', (299, 299), color=(73, 109, 137))
        
        # Save image to temp file
        temp_path = 'temp_test_image.jpg'
        img.save(temp_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(temp_path))
        
        # Clean up
        os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()