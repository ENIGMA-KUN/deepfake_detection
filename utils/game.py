import os
import random
import time
from pathlib import Path
import glob
import sys
import json
import numpy as np
from PIL import Image

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.detector import DeepfakeDetector

class DeepfakeGame:
    def __init__(self, detector=None, questions_per_round=config.GAME_CONFIG["questions_per_round"]):
        """
        Initialize the deepfake detection game
        
        Args:
            detector (DeepfakeDetector): Detector object to use for AI predictions
            questions_per_round (int): Number of questions per game round
        """
        self.detector = detector if detector else DeepfakeDetector()
        self.questions_per_round = questions_per_round
        self.current_round = {}
        self.game_history = []
        self.current_question_idx = 0
        self.game_state = {}
        
        # Load or create game database
        self._load_or_create_game_database()
    
    def _load_or_create_game_database(self):
        """Load existing game images or create a sample database if none exists"""
        # Check if game images directories exist and have images
        real_images = glob.glob(os.path.join(config.REAL_IMAGES_DIR, "*.[jp][pn]g"))
        fake_images = glob.glob(os.path.join(config.FAKE_IMAGES_DIR, "*.[jp][pn]g"))
        
        # If there are not enough images for a game, create a basic set
        if len(real_images) < self.questions_per_round or len(fake_images) < self.questions_per_round:
            print(f"Not enough game images found. Creating sample game database.")
            self._create_sample_game_database()
            
            # Refresh the image lists
            real_images = glob.glob(os.path.join(config.REAL_IMAGES_DIR, "*.[jp][pn]g"))
            fake_images = glob.glob(os.path.join(config.FAKE_IMAGES_DIR, "*.[jp][pn]g"))
        
        self.real_images = real_images
        self.fake_images = fake_images
        print(f"Game database loaded: {len(self.real_images)} real images, {len(self.fake_images)} fake images")
    
    def _create_sample_game_database(self):
        """Create a sample set of real and fake images for the game"""
        # This is a placeholder. In a real application, you would have a set of
        # pre-verified real and fake images for the game.
        # For demonstration purposes, we'll create basic colored rectangles.
        
        os.makedirs(config.REAL_IMAGES_DIR, exist_ok=True)
        os.makedirs(config.FAKE_IMAGES_DIR, exist_ok=True)
        
        # Create sample real images (green rectangles with faces)
        for i in range(20):
            img = Image.new('RGB', (300, 300), color=(100, 200, 100))
            # Add simple face-like features
            img.save(os.path.join(config.REAL_IMAGES_DIR, f"sample_real_{i}.jpg"))
        
        # Create sample fake images (red rectangles with faces)
        for i in range(20):
            img = Image.new('RGB', (300, 300), color=(200, 100, 100))
            # Add simple face-like features
            img.save(os.path.join(config.FAKE_IMAGES_DIR, f"sample_fake_{i}.jpg"))
        
        print(f"Created sample game database with 20 real and 20 fake images")
    
    def new_game(self):
        """Start a new game round"""
        # Reset current game
        self.current_round = {
            "questions": [],
            "user_score": 0,
            "ai_score": 0,
            "total_questions": self.questions_per_round,
            "start_time": time.time(),
        }
        self.current_question_idx = 0
        
        # Generate questions (balance real and fake)
        real_images = random.sample(self.real_images, self.questions_per_round // 2)
        fake_images = random.sample(self.fake_images, self.questions_per_round - len(real_images))
        
        all_images = real_images + fake_images
        random.shuffle(all_images)
        
        for img_path in all_images:
            # Determine true label based on directory
            is_fake = "fake" in os.path.dirname(img_path).lower()
            true_label = "fake" if is_fake else "real"
            
            # Get AI prediction
            try:
                result = self.detector.predict(img_path)
                ai_prediction = result["prediction"]
                confidence = result["confidence"]
            except Exception as e:
                print(f"Error predicting {img_path}: {str(e)}")
                ai_prediction = "unknown"
                confidence = 0.0
            
            # Add question to current round
            self.current_round["questions"].append({
                "image_path": img_path,
                "true_label": true_label,
                "ai_prediction": ai_prediction,
                "ai_confidence": confidence,
                "user_answer": None,
                "is_correct": None,
                "ai_is_correct": ai_prediction == true_label,
                "response_time": None,
            })
        
        return self.get_question()
    
    def get_question(self):
        """Get current question"""
        if self.current_question_idx >= len(self.current_round["questions"]):
            return None
        
        return {
            "question_number": self.current_question_idx + 1,
            "total_questions": self.questions_per_round,
            "image_path": self.current_round["questions"][self.current_question_idx]["image_path"],
        }
    
    def submit_answer(self, answer):
        """
        Submit user answer for current question
        
        Args:
            answer (str): User answer ('real' or 'fake')
            
        Returns:
            dict: Result of the answer and next question if available
        """
        if self.current_question_idx >= len(self.current_round["questions"]):
            return {"error": "No active question"}
        
        # Record user answer
        question = self.current_round["questions"][self.current_question_idx]
        question["user_answer"] = answer
        question["is_correct"] = answer == question["true_label"]
        question["response_time"] = time.time() - self.current_round["start_time"]
        
        # Update scores
        if question["is_correct"]:
            self.current_round["user_score"] += 1
        if question["ai_is_correct"]:
            self.current_round["ai_score"] += 1
        
        # Prepare result
        result = {
            "is_correct": question["is_correct"],
            "true_label": question["true_label"],
            "ai_prediction": question["ai_prediction"],
            "ai_confidence": question["ai_confidence"],
            "ai_is_correct": question["ai_is_correct"],
            "user_score": self.current_round["user_score"],
            "ai_score": self.current_round["ai_score"],
            "question_number": self.current_question_idx + 1,
            "total_questions": self.questions_per_round,
        }
        
        # Move to next question
        self.current_question_idx += 1
        
        # Check if game is complete
        if self.current_question_idx >= len(self.current_round["questions"]):
            # Calculate final scores and stats
            result["game_complete"] = True
            result["final_user_score"] = self.current_round["user_score"]
            result["final_ai_score"] = self.current_round["ai_score"]
            result["winner"] = "user" if self.current_round["user_score"] > self.current_round["ai_score"] else \
                               "ai" if self.current_round["ai_score"] > self.current_round["user_score"] else "tie"
            
            # Add to game history
            self.game_history.append(self.current_round)
        else:
            # Get next question
            result["next_question"] = self.get_question()
            result["game_complete"] = False
        
        return result
    
    def get_game_stats(self):
        """Get statistics for all played games"""
        if not self.game_history:
            return {
                "total_games": 0,
                "user_wins": 0,
                "ai_wins": 0,
                "ties": 0,
                "avg_user_score": 0,
                "avg_ai_score": 0,
            }
        
        total_games = len(self.game_history)
        user_wins = sum(1 for game in self.game_history if game["user_score"] > game["ai_score"])
        ai_wins = sum(1 for game in self.game_history if game["ai_score"] > game["user_score"])
        ties = total_games - user_wins - ai_wins
        
        avg_user_score = sum(game["user_score"] for game in self.game_history) / total_games
        avg_ai_score = sum(game["ai_score"] for game in self.game_history) / total_games
        
        return {
            "total_games": total_games,
            "user_wins": user_wins,
            "ai_wins": ai_wins,
            "ties": ties,
            "avg_user_score": avg_user_score,
            "avg_ai_score": avg_ai_score,
        }