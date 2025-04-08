import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
EFFICIENTNET_PATH = os.path.join(MODEL_DIR, "efficientnet_model.pth")
XCEPTION_PATH = os.path.join(MODEL_DIR, "xception_model.pth")

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
GAME_IMAGES_DIR = os.path.join(DATA_DIR, "game_images")
REAL_IMAGES_DIR = os.path.join(GAME_IMAGES_DIR, "real")
FAKE_IMAGES_DIR = os.path.join(GAME_IMAGES_DIR, "fake")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")

# Model configuration
MODEL_CONFIG = {
    "efficientnet": {
        "name": "efficientnet-b4",
        "input_size": 224,
    },
    "xception": {
        "input_size": 299,
    }
}

# Default model to use
DEFAULT_MODEL = "efficientnet"

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

# Game configuration
GAME_CONFIG = {
    "questions_per_round": 10,
    "time_limit": 10,  # seconds per question
}

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, GAME_IMAGES_DIR, REAL_IMAGES_DIR, FAKE_IMAGES_DIR, SAMPLE_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)