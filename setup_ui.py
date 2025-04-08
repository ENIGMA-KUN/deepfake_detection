#!/usr/bin/env python
"""
Setup script for the Deepfake Detection Platform UI
"""

import os
import shutil
import sys

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def copy_file(source, destination):
    """Copy a file from source to destination"""
    try:
        shutil.copyfile(source, destination)
        print(f"Copied {source} to {destination}")
    except Exception as e:
        print(f"Error copying {source} to {destination}: {e}")

def main():
    """Main function"""
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create necessary directories
    static_dir = os.path.join(project_root, "src", "static")
    create_directory_if_not_exists(static_dir)
    
    uploads_dir = os.path.join(project_root, "uploads")
    create_directory_if_not_exists(uploads_dir)
    
    # Copy index.html to static directory
    index_source = os.path.join(project_root, "src", "static", "index_template.html")
    index_dest = os.path.join(static_dir, "index.html")
    if os.path.exists(index_source):
        copy_file(index_source, index_dest)
    else:
        print(f"Warning: Source file {index_source} not found.")
    
    print("UI setup complete! You can now start the server with 'python run.py api'")

if __name__ == "__main__":
    main()