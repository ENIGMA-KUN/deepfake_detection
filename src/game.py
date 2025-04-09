import os
import random
from PIL import Image
import streamlit as st
import numpy as np

class DeepfakeGame:
    """
    Game mode for deepfake detection
    """
    def __init__(self, detector, real_images_dir='assets/sample_images/real', fake_images_dir='assets/sample_images/fake'):
        self.detector = detector
        self.real_images_dir = real_images_dir
        self.fake_images_dir = fake_images_dir
        self.score = 0
        self.total_questions = 0
        
        # Create directories if they don't exist
        os.makedirs(real_images_dir, exist_ok=True)
        os.makedirs(fake_images_dir, exist_ok=True)
        
        # Load all image paths at initialization
        self.real_images = self.get_images(self.real_images_dir)
        self.fake_images = self.get_images(self.fake_images_dir)
        self.all_images = self.real_images + self.fake_images
        
        # Create mapping of image paths to labels
        self.image_labels = {}
        for img in self.real_images:
            self.image_labels[img] = False  # False = Real
        for img in self.fake_images:
            self.image_labels[img] = True   # True = Fake
    
    def get_images(self, directory):
        """
        Get list of images from directory
        """
        if not os.path.exists(directory):
            return []
            
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def get_all_images(self):
        """
        Return all available images
        """
        return self.all_images, self.image_labels
    
    def evaluate_image(self, image_path):
        """
        Get the actual label and model prediction for an image
        """
        # Get actual label (whether the image is fake)
        actual_is_fake = self.image_labels.get(image_path, None)
        
        if actual_is_fake is None:
            return None
            
        # Run the detector to get model prediction
        predictions, marked_image = self.detector.predict(image_path=image_path)
        
        # If there are faces in the image
        if predictions:
            model_prediction = predictions[0][1]  # is_fake from the first face
            model_confidence = predictions[0][2]  # confidence from the first face
        else:
            model_prediction = False
            model_confidence = 0.5
            
        return {
            'image_path': image_path,
            'actual_is_fake': actual_is_fake,
            'model_prediction': model_prediction,
            'model_confidence': model_confidence,
            'marked_image': marked_image
        }
    
    def evaluate_answer(self, image_path, user_answer):
        """
        Evaluate user's answer and update score
        """
        # Get actual label (whether the image is fake)
        actual_is_fake = self.image_labels.get(image_path, None)
        
        if actual_is_fake is None:
            return None
            
        is_correct = user_answer == actual_is_fake
        
        if is_correct:
            self.score += 1
            
        self.total_questions += 1
        
        # Run the detector to get model prediction
        result = self.evaluate_image(image_path)
        if not result:
            return None
            
        return {
            'is_correct': is_correct,
            'score': self.score,
            'total': self.total_questions,
            'accuracy': (self.score / self.total_questions * 100) if self.total_questions > 0 else 0,
            'actual_is_fake': actual_is_fake,
            'model_prediction': result['model_prediction'],
            'model_confidence': result['model_confidence'],
            'marked_image': result['marked_image']
        }
        
    def reset_game(self):
        """
        Reset the game score
        """
        self.score = 0
        self.total_questions = 0