#!/usr/bin/env python3
"""
Project initialization script for DeepFake Detection Platform.
This script creates the necessary directory structure and downloads sample data.
"""

import os
import sys
import shutil
from pathlib import Path
import requests
import zipfile
import io
import random
from PIL import Image, ImageDraw
import numpy as np

def create_directory_structure():
    """Create the project directory structure"""
    print("Creating directory structure...")
    
    # Base directories
    directories = [
        "models",
        "data/game_images/real",
        "data/game_images/fake",
        "data/sample_images",
        "utils",
        "assets",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}/")
    
    print("Directory structure created successfully!")

def create_init_file():
    """Create __init__.py file for utils module"""
    init_path = os.path.join("utils", "__init__.py")
    
    with open(init_path, "w") as f:
        f.write("# Init file for utils module\n")
    
    print(f"Created: {init_path}")

def create_sample_images():
    """Create sample real and fake face images for testing"""
    print("Creating sample images...")
    
    # Paths
    real_dir = "data/game_images/real"
    fake_dir = "data/game_images/fake"
    sample_dir = "data/sample_images"
    
    # Create sample real images
    for i in range(10):
        # Create a blank image with skin tone background
        img = Image.new('RGB', (300, 300), color=(random.randint(200, 240), 
                                                 random.randint(170, 210), 
                                                 random.randint(140, 180)))
        draw = ImageDraw.Draw(img)
        
        # Draw a face oval
        draw.ellipse([(75, 50), (225, 230)], fill=(random.randint(220, 255), 
                                                  random.randint(190, 220), 
                                                  random.randint(160, 190)))
        
        # Draw eyes
        eye_color = (random.randint(10, 80), random.randint(10, 80), random.randint(10, 80))
        draw.ellipse([(115, 110), (135, 125)], fill='white', outline='black')
        draw.ellipse([(165, 110), (185, 125)], fill='white', outline='black')
        draw.ellipse([(120, 113), (130, 123)], fill=eye_color)
        draw.ellipse([(170, 113), (180, 123)], fill=eye_color)
        
        # Draw mouth
        draw.arc([(125, 150), (175, 190)], 0, 180, fill='black', width=2)
        
        # Save the image
        img.save(os.path.join(real_dir, f"sample_real_{i}.jpg"))
        
        # Also save a couple as sample images
        if i < 2:
            img.save(os.path.join(sample_dir, f"sample_real_{i}.jpg"))
    
    # Create sample fake images
    for i in range(10):
        # Create a blank image with slightly unnatural skin tone
        img = Image.new('RGB', (300, 300), color=(random.randint(200, 240), 
                                                 random.randint(150, 190), 
                                                 random.randint(150, 170)))
        draw = ImageDraw.Draw(img)
        
        # Draw a face oval with unusual color
        draw.ellipse([(75, 50), (225, 230)], fill=(random.randint(220, 255), 
                                                  random.randint(180, 210), 
                                                  random.randint(170, 200)))
        
        # Draw asymmetric eyes
        eye_color = (random.randint(30, 100), random.randint(30, 100), random.randint(80, 150))
        draw.ellipse([(110, 105), (135, 125)], fill='white', outline='black')
        draw.ellipse([(165, 115), (190, 130)], fill='white', outline='black')
        draw.ellipse([(115, 108), (128, 123)], fill=eye_color)
        draw.ellipse([(170, 118), (185, 128)], fill=eye_color)
        
        # Draw mouth (slightly off)
        draw.arc([(125, 160), (175, 200)], 0, 180, fill='black', width=2)
        
        # Add some artifacts/noise
        for _ in range(20):
            x = random.randint(0, 299)
            y = random.randint(0, 299)
            radius = random.randint(1, 3)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color)
        
        # Save the image
        img.save(os.path.join(fake_dir, f"sample_fake_{i}.jpg"))
        
        # Also save a couple as sample images
        if i < 2:
            img.save(os.path.join(sample_dir, f"sample_fake_{i}.jpg"))
    
    print(f"Created 10 sample real images in {real_dir}/")
    print(f"Created 10 sample fake images in {fake_dir}/")
    print(f"Created 4 sample images in {sample_dir}/")

def main():
    """Main function to initialize the project"""
    print("Initializing DeepFake Detection Platform project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__.py file
    create_init_file()
    
    # Create sample images
    create_sample_images()
    
    print("\nProject initialization completed successfully!")
    print("\nTo run the application:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Start the application: streamlit run app.py")

if __name__ == "__main__":
    main()