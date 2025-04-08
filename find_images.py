import os

def list_images(directory):
    """List all image files in a directory (recursively)"""
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                images.append(full_path)
    
    return images

# Base directory to search
base_dir = 'data'
print(f"Searching for images in {os.path.abspath(base_dir)}...")

# Find all images
images = list_images(base_dir)

# Print results
if images:
    print(f"Found {len(images)} images:")
    for i, image in enumerate(images[:20]):  # Print first 20 to avoid overwhelming output
        print(f"{i+1}. {image}")
    
    if len(images) > 20:
        print(f"... and {len(images) - 20} more")
else:
    print("No images found!")