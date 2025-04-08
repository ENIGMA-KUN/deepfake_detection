import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import torch
from torchvision import transforms
import shutil

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Paths
FACEFORENSICS_PATH = config['paths']['faceforensics']
CELEBDF_PATH = config['paths']['celebdf']
PROCESSED_IMAGES_PATH = config['paths']['processed_images']

def ensure_directory(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_face(image_path, target_size=(299, 299)):
    """
    Extract face from image using simple method (for testing)
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Size to resize the face to
    
    Returns:
        PIL.Image or None: Extracted face or None if no face detected
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # For testing, just use the central region of the image as a "face"
        h, w = image_rgb.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 2
        
        # Extract face region
        face = image_rgb[
            max(0, center_y - size):min(h, center_y + size),
            max(0, center_x - size):min(w, center_x + size)
        ]
        
        # Convert to PIL image and resize
        pil_face = Image.fromarray(face).resize(target_size)
        return pil_face
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_dataset(dataset_path, output_path, target_size=(299, 299), 
                       label=None, limit=None):
    """
    Preprocess a dataset by extracting faces and saving them
    
    Args:
        dataset_path (str): Path to dataset
        output_path (str): Path to save processed images
        target_size (tuple): Size to resize faces to
        label (int): Label for the dataset (0 for real, 1 for fake)
        limit (int): Limit the number of processed images
    """
    # Ensure output directory exists
    ensure_directory(output_path)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if limit:
        image_files = image_files[:limit]
    
    # Create a metadata file
    metadata_path = os.path.join(output_path, 'metadata.txt')
    
    # Process images
    print(f"Processing {len(image_files)} images from {dataset_path}")
    processed_count = 0
    
    with open(metadata_path, 'w') as metadata_file:
        for image_path in tqdm(image_files):
            # Extract face
            face = extract_face(image_path, target_size=target_size)
            
            if face:
                # Create a unique filename
                filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{processed_count}.jpg"
                output_file = os.path.join(output_path, filename)
                
                # Save the face
                face.save(output_file)
                
                # Write metadata
                metadata_file.write(f"{filename},{label if label is not None else 'unknown'}\n")
                
                processed_count += 1
    
    print(f"Processed {processed_count} images. Saved to {output_path}")
    return processed_count

def process_faceforensics(limit=1000):
    """Process FaceForensics++ dataset"""
    # Real images
    real_path = os.path.join(FACEFORENSICS_PATH, 'original_sequences', 'youtube', 'c23', 'frames')
    real_output = os.path.join(PROCESSED_IMAGES_PATH, 'faceforensics_real')
    
    # Fake images (DeepFakes)
    fake_path = os.path.join(FACEFORENSICS_PATH, 'manipulated_sequences', 'Deepfakes', 'c23', 'frames')
    fake_output = os.path.join(PROCESSED_IMAGES_PATH, 'faceforensics_fake')
    
    # Process datasets
    real_count = preprocess_dataset(real_path, real_output, label=0, limit=limit)
    fake_count = preprocess_dataset(fake_path, fake_output, label=1, limit=limit)
    
    print(f"Processed {real_count} real and {fake_count} fake images from FaceForensics++")

def process_celebdf(limit=1000):
    """Process Celeb-DF dataset"""
    # Real images
    real_path = os.path.join(CELEBDF_PATH, 'Celeb-real')
    real_output = os.path.join(PROCESSED_IMAGES_PATH, 'celebdf_real')
    
    # Fake images
    fake_path = os.path.join(CELEBDF_PATH, 'Celeb-synthesis')
    fake_output = os.path.join(PROCESSED_IMAGES_PATH, 'celebdf_fake')
    
    # Process datasets
    real_count = preprocess_dataset(real_path, real_output, label=0, limit=limit)
    fake_count = preprocess_dataset(fake_path, fake_output, label=1, limit=limit)
    
    print(f"Processed {real_count} real and {fake_count} fake images from Celeb-DF")

def combine_datasets():
    """Combine all processed datasets into train/val/test splits"""
    # Paths
    combined_path = os.path.join(PROCESSED_IMAGES_PATH, 'combined')
    train_path = os.path.join(combined_path, 'train')
    val_path = os.path.join(combined_path, 'val')
    test_path = os.path.join(combined_path, 'test')
    
    # Create directories
    for path in [train_path, val_path, test_path]:
        ensure_directory(path)
        ensure_directory(os.path.join(path, 'real'))
        ensure_directory(os.path.join(path, 'fake'))
    
    # Datasets to combine
    real_dirs = [
        os.path.join(PROCESSED_IMAGES_PATH, 'faceforensics_real'),
        os.path.join(PROCESSED_IMAGES_PATH, 'celebdf_real')
    ]
    
    fake_dirs = [
        os.path.join(PROCESSED_IMAGES_PATH, 'faceforensics_fake'),
        os.path.join(PROCESSED_IMAGES_PATH, 'celebdf_fake')
    ]
    
    # Split ratios
    train_ratio, val_ratio = 0.7, 0.15  # test_ratio = 0.15
    
    # Process real images
    real_files = []
    for dir_path in real_dirs:
        if os.path.exists(dir_path):
            metadata_path = os.path.join(dir_path, 'metadata.txt')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        filename, _ = line.strip().split(',')
                        real_files.append((os.path.join(dir_path, filename), 'real'))
    
    # Process fake images
    fake_files = []
    for dir_path in fake_dirs:
        if os.path.exists(dir_path):
            metadata_path = os.path.join(dir_path, 'metadata.txt')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        filename, _ = line.strip().split(',')
                        fake_files.append((os.path.join(dir_path, filename), 'fake'))
    
    # Shuffle files
    np.random.shuffle(real_files)
    np.random.shuffle(fake_files)
    
    # Split real files
    n_real = len(real_files)
    n_train_real = int(n_real * train_ratio)
    n_val_real = int(n_real * val_ratio)
    
    real_train = real_files[:n_train_real]
    real_val = real_files[n_train_real:n_train_real+n_val_real]
    real_test = real_files[n_train_real+n_val_real:]
    
    # Split fake files
    n_fake = len(fake_files)
    n_train_fake = int(n_fake * train_ratio)
    n_val_fake = int(n_fake * val_ratio)
    
    fake_train = fake_files[:n_train_fake]
    fake_val = fake_files[n_train_fake:n_train_fake+n_val_fake]
    fake_test = fake_files[n_train_fake+n_val_fake:]
    
    # Copy files
    def copy_files(files, dest_base):
        for src_path, label in tqdm(files):
            if os.path.exists(src_path):
                dest_path = os.path.join(dest_base, label, os.path.basename(src_path))
                shutil.copy2(src_path, dest_path)
    
    print("Copying training files...")
    copy_files(real_train, train_path)
    copy_files(fake_train, train_path)
    
    print("Copying validation files...")
    copy_files(real_val, val_path)
    copy_files(fake_val, val_path)
    
    print("Copying test files...")
    copy_files(real_test, test_path)
    copy_files(fake_test, test_path)
    
    # Create dataset summary
    summary = {
        'total': {
            'real': n_real,
            'fake': n_fake,
            'total': n_real + n_fake
        },
        'train': {
            'real': len(real_train),
            'fake': len(fake_train),
            'total': len(real_train) + len(fake_train)
        },
        'val': {
            'real': len(real_val),
            'fake': len(fake_val),
            'total': len(real_val) + len(fake_val)
        },
        'test': {
            'real': len(real_test),
            'fake': len(fake_test),
            'total': len(real_test) + len(fake_test)
        }
    }
    
    # Save summary
    with open(os.path.join(combined_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Dataset combined and split. Summary saved to {os.path.join(combined_path, 'summary.json')}")
    
    return summary

if __name__ == "__main__":
    # Create necessary directories
    ensure_directory(PROCESSED_IMAGES_PATH)
    
    # Process FaceForensics++ if available
    if os.path.exists(FACEFORENSICS_PATH):
        process_faceforensics()
    else:
        print(f"FaceForensics++ dataset not found at {FACEFORENSICS_PATH}")
    
    # Process Celeb-DF if available
    if os.path.exists(CELEBDF_PATH):
        process_celebdf()
    else:
        print(f"Celeb-DF dataset not found at {CELEBDF_PATH}")
    
    # Combine datasets
    combine_datasets()