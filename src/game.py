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
        self.current_image = None
        self.current_is_fake = None
        self.score = 0
        self.total_questions = 0
        
        # Create directories if they don't exist
        os.makedirs(real_images_dir, exist_ok=True)
        os.makedirs(fake_images_dir, exist_ok=True)
        
        # Check if sample images are available
        self.check_sample_images()
    
    def check_sample_images(self):
        """
        Check if sample images are available, download them if not
        """
        real_images = self.get_images(self.real_images_dir)
        fake_images = self.get_images(self.fake_images_dir)
        
        if len(real_images) < 10 or len(fake_images) < 10:
            st.warning("Sample images for the game mode are missing. Please add at least 10 real and 10 fake images to the assets/sample_images directory.")
            st.info("You can download sample images from datasets like FaceForensics++, CelebDF, or use the DF40 dataset.")
    
    def get_images(self, directory):
        """
        Get list of images from directory
        """
        if not os.path.exists(directory):
            return []
            
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def get_new_question(self):
        """
        Generate a new question with a random image
        """
        real_images = self.get_images(self.real_images_dir)
        fake_images = self.get_images(self.fake_images_dir)
        
        if not real_images or not fake_images:
            st.error("No sample images found for the game. Please add images to the assets/sample_images directory.")
            return None, None
        
        # Randomly choose if the next image is real or fake
        is_fake = random.choice([True, False])
        
        # Get image path based on real/fake choice
        if is_fake:
            self.current_image = random.choice(fake_images)
            self.current_is_fake = True
        else:
            self.current_image = random.choice(real_images)
            self.current_is_fake = False
            
        return self.current_image, self.current_is_fake
    
    def evaluate_answer(self, user_answer):
        """
        Evaluate user's answer and update score
        """
        if self.current_is_fake is None:
            return None
            
        is_correct = user_answer == self.current_is_fake
        
        if is_correct:
            self.score += 1
            
        self.total_questions += 1
        
        # Run the detector to get model prediction
        predictions, marked_image = self.detector.predict(image_path=self.current_image)
        
        # If there are faces in the image
        if predictions:
            model_prediction = predictions[0][1]  # is_fake from the first face
            model_confidence = predictions[0][2]  # confidence from the first face
        else:
            model_prediction = False
            model_confidence = 0.5
            
        return {
            'is_correct': is_correct,
            'score': self.score,
            'total': self.total_questions,
            'accuracy': (self.score / self.total_questions * 100) if self.total_questions > 0 else 0,
            'model_prediction': model_prediction,
            'model_confidence': model_confidence,
            'marked_image': marked_image
        }
        
    def reset_game(self):
        """
        Reset the game score
        """
        self.score = 0
        self.total_questions = 0