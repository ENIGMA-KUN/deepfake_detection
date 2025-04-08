from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import uuid
import shutil
import numpy as np
from PIL import Image
import io
import tempfile
import sys
import time
from typing import Optional, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import detector
from src.models.image.inference import DeepfakeDetector

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Create upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize app
app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfakes in images, audio, and video",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
image_detector_xception = None
image_detector_efficientnet = None

def get_image_detector(model_type='xception'):
    """Get image detector instance"""
    global image_detector_xception, image_detector_efficientnet
    
    if model_type == 'xception':
        if image_detector_xception is None:
            try:
                image_detector_xception = DeepfakeDetector(model_type='xception')
            except Exception as e:
                print(f"Error initializing Xception detector: {str(e)}")
                return None
        return image_detector_xception
    elif model_type == 'efficientnet':
        if image_detector_efficientnet is None:
            try:
                image_detector_efficientnet = DeepfakeDetector(model_type='efficientnet')
            except Exception as e:
                print(f"Error initializing EfficientNet detector: {str(e)}")
                return None
        return image_detector_efficientnet
    else:
        return None

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Deepfake Detection API"}

@app.get("/status")
def get_status():
    """Get API status"""
    return {
        "status": "online",
        "models": {
            "image": {
                "xception": image_detector_xception is not None,
                "efficientnet": image_detector_efficientnet is not None
            }
        }
    }

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model_type: str = 'xception',
    return_visualization: bool = False
):
    """
    Detect deepfakes in an image
    
    Args:
        file: Image file
        model_type: Model type ('xception' or 'efficientnet')
        return_visualization: Whether to return visualization
    """
    if model_type not in ['xception', 'efficientnet']:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be 'xception' or 'efficientnet'.")
    
    # Get file content
    content = await file.read()
    
    # Create a unique filename
    filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(content)
    
    # Get detector
    detector = get_image_detector(model_type)
    if detector is None:
        raise HTTPException(status_code=500, detail="Failed to initialize detector")
    
    try:
        # Detect
        result = detector.predict(filepath, return_heatmap=return_visualization)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create result
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "fake_probability": result['fake_probability'],
            "real_probability": result['real_probability']
        }
        
        # Add visualization if requested
        if return_visualization and 'heatmap' in result:
            # Save visualization
            vis_filename = f"{os.path.splitext(filename)[0]}_vis.jpg"
            vis_filepath = os.path.join(RESULTS_DIR, vis_filename)
            
            # Create visualization
            fig = detector.visualize_prediction(filepath, vis_filepath)
            
            # Add visualization path to response
            response["visualization"] = f"/results/{vis_filename}"
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return response
    
    except Exception as e:
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{filename}")
def get_result(filename: str):
    """Get result file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    return FileResponse(filepath)

@app.get("/models/image")
def get_image_models():
    """Get available image models"""
    models_dir = os.path.join('models', 'image')
    if not os.path.exists(models_dir):
        return {"models": []}
    
    # Get subdirectories
    models = []
    for model_type in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_type)
        if os.path.isdir(model_dir):
            checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            models.append({
                "type": model_type,
                "checkpoints": checkpoints
            })
    
    return {"models": models}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Deepfake Detection API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    
    args = parser.parse_args()
    
    # Pre-load models
    print("Initializing models...")
    get_image_detector('xception')
    get_image_detector('efficientnet')
    
    # Run API
    uvicorn.run(app, host=args.host, port=args.port)