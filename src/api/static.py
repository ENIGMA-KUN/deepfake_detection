from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import os

def configure_static_files(app: FastAPI):
    """
    Configure static files for the FastAPI application
    
    Args:
        app: FastAPI application
    """
    # Determine the static files directory
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "static")
    
    # Check if the directory exists
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        
    # Mount the static files
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    return static_dir