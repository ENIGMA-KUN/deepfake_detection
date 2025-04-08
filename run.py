import os
import argparse
import subprocess
import sys

def setup_environment():
    """Check and install required packages"""
    try:
        import numpy
        import torch
        import cv2
        print("Core dependencies already installed")
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_dataset():
    """Download and prepare dataset"""
    print("Setting up datasets...")
    subprocess.check_call([sys.executable, "src/data/download_datasets.py", "--synthetic"])
    
    print("Processing datasets...")
    subprocess.check_call([sys.executable, "src/data/preprocess_images.py"])

def train_model(model_type, epochs, batch_size, learning_rate):
    """Train model"""
    print(f"Training {model_type} model...")
    
    if model_type == 'resnet':
        script_path = "train_resnet.py"
    else:
        script_path = "src/models/image/train.py"
        
    cmd = [
        sys.executable, script_path,
        "--model", model_type,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate)
    ]
    subprocess.check_call(cmd)

def run_inference(model_type, image_path, output_path):
    """Run inference on an image"""
    print(f"Running inference with {model_type} model...")
    cmd = [
        sys.executable, "src/models/image/inference.py",
        "--model", model_type,
        "--image", image_path
    ]
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    subprocess.check_call(cmd)

def run_api(host, port):
    """Run API server"""
    print(f"Starting API server on {host}:{port}...")
    cmd = [
        sys.executable, "src/api/app.py",
        "--host", host,
        "--port", str(port)
    ]
    subprocess.check_call(cmd)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Platform')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project')
    setup_parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset download and preprocessing')
    
    # Train command
   # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model', type=str, default='resnet', choices=['xception', 'efficientnet', 'resnet'],
                         help='Model type')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--model', type=str, default='xception', choices=['xception', 'efficientnet'],
                                 help='Model type')
    inference_parser.add_argument('--image', type=str, required=True, help='Input image path')
    inference_parser.add_argument('--output', type=str, help='Output visualization path')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Run API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    api_parser.add_argument('--port', type=int, default=8000, help='Port')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'setup':
        setup_environment()
        if not args.skip_dataset:
            setup_dataset()
    
    elif args.command == 'train':
        train_model(args.model, args.epochs, args.batch_size, args.lr)
    
    elif args.command == 'inference':
        run_inference(args.model, args.image, args.output)
    
    elif args.command == 'api':
        run_api(args.host, args.port)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()