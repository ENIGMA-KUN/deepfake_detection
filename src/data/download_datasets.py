import os
import argparse
import requests
import zipfile
import tarfile
import gdown
import shutil
from tqdm import tqdm
import json
import time
from PIL import Image, ImageDraw

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

def download_file(url, destination):
    """
    Download a file from URL to destination with progress bar
    
    Args:
        url (str): URL to download
        destination (str): Destination path
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))

def extract_zip(zip_path, extract_path):
    """
    Extract a zip file
    
    Args:
        zip_path (str): Path to zip file
        extract_path (str): Path to extract to
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        
        with tqdm(total=total_files, desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_path)
                pbar.update(1)

def create_dummy_image(path, width=300, height=300, color=(255, 200, 150)):
    """
    Create a dummy image for testing
    
    Args:
        path (str): Path to save image
        width (int): Image width
        height (int): Image height
        color (tuple): RGB color
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Create a colored image
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Face outline
    draw.ellipse([(width//4, height//4), (3*width//4, 3*height//4)], fill=(255, 220, 200))
    
    # Eyes
    eye_size = width // 10
    draw.ellipse([(width//3 - eye_size//2, height//3 - eye_size//2), 
                 (width//3 + eye_size//2, height//3 + eye_size//2)], fill=(255, 255, 255))
    draw.ellipse([(2*width//3 - eye_size//2, height//3 - eye_size//2), 
                 (2*width//3 + eye_size//2, height//3 + eye_size//2)], fill=(255, 255, 255))
    
    # Pupils
    pupil_size = eye_size // 2
    draw.ellipse([(width//3 - pupil_size//2, height//3 - pupil_size//2), 
                 (width//3 + pupil_size//2, height//3 + pupil_size//2)], fill=(0, 0, 0))
    draw.ellipse([(2*width//3 - pupil_size//2, height//3 - pupil_size//2), 
                 (2*width//3 + pupil_size//2, height//3 + pupil_size//2)], fill=(0, 0, 0))
    
    # Mouth
    draw.arc([(width//3, 2*height//3 - height//10), (2*width//3, 2*height//3 + height//10)], 
            0, 180, fill=(150, 50, 50), width=5)
    
    # Save image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def create_synthetic_dataset():
    """
    Create synthetic dataset with dummy face images
    """
    print("Creating synthetic dataset with dummy face images...")
    
    # Base paths
    faceforensics_path = os.path.join(config['data_path']['raw'], "faceforensics")
    celebdf_path = os.path.join(config['data_path']['raw'], "celebdf")
    
    # Create FaceForensics++ structure
    real_frames_dir = os.path.join(faceforensics_path, "original_sequences", "youtube", "c23", "frames")
    fake_frames_dir = os.path.join(faceforensics_path, "manipulated_sequences", "Deepfakes", "c23", "frames")
    
    os.makedirs(real_frames_dir, exist_ok=True)
    os.makedirs(fake_frames_dir, exist_ok=True)
    
    # Create Celeb-DF structure
    celeb_real_dir = os.path.join(celebdf_path, "Celeb-real")
    celeb_fake_dir = os.path.join(celebdf_path, "Celeb-synthesis")
    
    os.makedirs(celeb_real_dir, exist_ok=True)
    os.makedirs(celeb_fake_dir, exist_ok=True)
    
    # Create images
    num_samples = 10
    
    # Create real samples (varying skin tones)
    real_colors = [
        (255, 220, 200),  # Light skin
        (240, 200, 180),  # Medium-light skin
        (220, 180, 160),  # Medium skin
        (200, 160, 140),  # Medium-dark skin
        (180, 140, 120)   # Dark skin
    ]
    
    for i in range(num_samples):
        color = real_colors[i % len(real_colors)]
        
        # FaceForensics++ real
        create_dummy_image(
            os.path.join(real_frames_dir, f"real_{i}.jpg"),
            color=color
        )
        
        # Celeb-DF real
        create_dummy_image(
            os.path.join(celeb_real_dir, f"real_{i}.jpg"),
            color=color
        )
    
    # Create fake samples (unusual colors to make them visually different)
    fake_colors = [
        (240, 240, 200),  # Yellowish
        (220, 200, 240),  # Purplish
        (200, 240, 220),  # Greenish
        (240, 200, 220),  # Pinkish
        (220, 220, 240)   # Bluish
    ]
    
    for i in range(num_samples):
        color = fake_colors[i % len(fake_colors)]
        
        # FaceForensics++ fake
        create_dummy_image(
            os.path.join(fake_frames_dir, f"fake_{i}.jpg"),
            color=color
        )
        
        # Celeb-DF fake
        create_dummy_image(
            os.path.join(celeb_fake_dir, f"fake_{i}.jpg"),
            color=color
        )
    
    print(f"Created synthetic datasets at {faceforensics_path} and {celebdf_path}")
    print("Note: These are dummy images for testing the pipeline")
    print("Replace with real datasets when available")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset instead of downloading')
    
    args = parser.parse_args()
    
    # Create synthetic dataset for testing
    create_synthetic_dataset()
    print("Dataset creation complete!")