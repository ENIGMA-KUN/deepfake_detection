import os
import shutil
import random
import argparse
from tqdm import tqdm

def create_directories():
    """Create necessary directories for sample images"""
    os.makedirs('assets/sample_images/real', exist_ok=True)
    os.makedirs('assets/sample_images/fake', exist_ok=True)
    print("Created sample image directories")

def find_image_files(directory):
    """Find all image files in a directory and its subdirectories"""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

def copy_random_images(source_dirs, destination_dir, num_images=20):
    """Copy random images from source directories to destination directory"""
    all_images = []
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            images = find_image_files(source_dir)
            all_images.extend(images)
            print(f"Found {len(images)} images in {source_dir}")
    
    if not all_images:
        print(f"No images found in the specified directories: {source_dirs}")
        return False
    
    # Randomly select images
    num_to_copy = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_copy)
    
    # Copy selected images
    print(f"Copying {num_to_copy} images to {destination_dir}...")
    for i, image_path in enumerate(tqdm(selected_images)):
        # Create a numbered filename to avoid collisions
        extension = os.path.splitext(image_path)[1]
        new_filename = f"sample_{i+1}{extension}"
        destination_path = os.path.join(destination_dir, new_filename)
        
        try:
            shutil.copy2(image_path, destination_path)
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
    
    print(f"Successfully copied {num_to_copy} images to {destination_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare sample images for DeepFake Detective')
    parser.add_argument('--df40_dir', type=str, help='Path to DF40 dataset directory', default=None)
    parser.add_argument('--real_sources', type=str, nargs='+', help='Directories containing real face images', default=[])
    parser.add_argument('--fake_sources', type=str, nargs='+', help='Directories containing fake face images', default=[])
    parser.add_argument('--num_images', type=int, help='Number of images to copy from each category', default=20)
    
    args = parser.parse_args()
    
    create_directories()
    
    real_sources = args.real_sources
    fake_sources = args.fake_sources
    
    # If DF40 directory is provided, use it to find real and fake images
    if args.df40_dir and os.path.exists(args.df40_dir):
        # Example paths for real images in DF40
        if not real_sources:
            real_sources = [
                os.path.join(args.df40_dir, 'real_images'),  # Adjust based on actual structure
                os.path.join(args.df40_dir, 'FF++_real')
            ]
        
        # Example paths for fake images in DF40
        if not fake_sources:
            fake_sources = [
                os.path.join(args.df40_dir, 'simswap', 'ff'),
                os.path.join(args.df40_dir, 'facedancer', 'ff'),
                os.path.join(args.df40_dir, 'blendface', 'ff')
            ]
    
    # Copy real images
    real_success = copy_random_images(
        real_sources, 
        'assets/sample_images/real', 
        args.num_images
    )
    
    # Copy fake images
    fake_success = copy_random_images(
        fake_sources, 
        'assets/sample_images/fake', 
        args.num_images
    )
    
    if real_success and fake_success:
        print("\n✅ Sample preparation complete! You can now run the application.")
    else:
        print("\n⚠️ Some images could not be copied. Please check the paths and try again.")
        print("You can manually add images to assets/sample_images/real and assets/sample_images/fake")

if __name__ == "__main__":
    main()