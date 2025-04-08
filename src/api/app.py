from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import tempfile
import shutil
import sys
import importlib.util

from src.api.static import configure_static_files

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import detector
from src.models.image.inference import DeepfakeDetector

# Configuration directory
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Initialize API
app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfakes in images, audio, and video",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
RESULTS_DIR = "uploads"
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "static")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Configure static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create templates directory
templates_dir = os.path.join(STATIC_DIR, "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Initialize detectors
image_detector_resnet = None
image_detector_efficientnet = None

def get_image_detector(model_type="resnet"):
    """Get image detector instance"""
    global image_detector_resnet, image_detector_efficientnet
    
    if model_type == "resnet":
        if image_detector_resnet is None:
            try:
                image_detector_resnet = DeepfakeDetector(model_type="resnet")
            except Exception as e:
                print(f"Error initializing ResNet detector: {str(e)}")
                return None
        return image_detector_resnet
    elif model_type == "efficientnet":
        if image_detector_efficientnet is None:
            try:
                image_detector_efficientnet = DeepfakeDetector(model_type="efficientnet")
            except Exception as e:
                print(f"Error initializing EfficientNet detector: {str(e)}")
                return None
        return image_detector_efficientnet
    else:
        return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Root endpoint that serves the frontend"""
    # Check if index.html exists in static directory
    index_path = os.path.join(STATIC_DIR, "index.html")
    
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        # If index.html doesn't exist, create it
        with open(index_path, "w") as f:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/index_template.html"), "r") as template:
                f.write(template.read())
                
        with open(index_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

@app.get("/status")
def get_status():
    """Get API status"""
    return {
        "status": "online",
        "models": {
            "image": {
                "resnet": image_detector_resnet is not None,
                "efficientnet": image_detector_efficientnet is not None
            }
        }
    }

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model_type: str = "resnet",
    return_visualization: bool = False
):
    """
    Detect deepfakes in an image
    
    Args:
        file: Image file
        model_type: Model type ('resnet' or 'efficientnet')
        return_visualization: Whether to return visualization
    """
    if model_type not in ["resnet", "efficientnet"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be 'resnet' or 'efficientnet'")
    
    # Get file content
    content = await file.read()
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        temp_file.write(content)
        filepath = temp_file.name
    
    # Use detector
    detector = get_image_detector(model_type)
    if detector is None:
        raise HTTPException(status_code=500, detail="Failed to initialize detector")
    
    try:
        # Predict
        result = detector.predict(filepath, return_heatmap=return_visualization)
        
        # Create result
        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "real_probability": result["real_probability"],
            "fake_probability": result["fake_probability"]
        }
        
        # Add visualization if requested
        if return_visualization and "heatmap" in result:
            # Save visualization
            vis_filename = f"{os.path.basename(file.filename).split('.')[0]}_vis.jpg"
            vis_filepath = os.path.join(RESULTS_DIR, vis_filename)
            
            # Create visualization
            result["heatmap"]["raw"].save(vis_filepath)
            
            # Add visualization url to response
            response["visualization"] = f"/results/{vis_filename}"
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
            
        return response
        
    except Exception as e:
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
    models_dir = os.path.join("models", "image")
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for model_type in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_type)
        if os.path.isdir(model_dir):
            checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            models.append({
                "type": model_type,
                "checkpoints": checkpoints
            })
    
    return {"models": models}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Deepfake Detection API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    
    args = parser.parse_args()
    
    # Initialize models
    print("Initializing models...")
    get_image_detector("resnet")
    
    # Run API
    uvicorn.run(app, host=args.host, port=args.port)