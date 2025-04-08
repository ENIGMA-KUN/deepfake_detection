# DeepFake Detection Platform - Implementation Guide

This guide provides step-by-step instructions for deploying and customizing the DeepFake Detection Platform.

## Quick Start

### 1. Environment Setup

Create a project directory and set up a virtual environment:

```bash
# Create project directory
mkdir deepfake-detection-app
cd deepfake-detection-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. File Structure Creation

Create the necessary directories and files according to the project structure:

```bash
# Create directories
mkdir -p models data/game_images/real data/game_images/fake data/sample_images utils assets

# Create empty files
touch app.py config.py requirements.txt README.md utils/__init__.py utils/detector.py utils/game.py assets/style.css
```

### 3. Copy Code Files

Copy the contents of each file from the provided implementations:

1. `requirements.txt`: Dependencies list
2. `config.py`: Configuration settings
3. `utils/detector.py`: DeepFake detection implementation
4. `utils/__init__.py`: Module initialization
5. `utils/game.py`: Game mode implementation
6. `assets/style.css`: Custom styling
7. `app.py`: Main Streamlit application

### 4. Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application should now be running at http://localhost:8501

## Advanced Configuration

### Using Pre-trained Models

The application is designed to work with pre-trained models. To use your own fine-tuned models:

1. Place your models in the `models` directory:
   - EfficientNet: `models/efficientnet_model.pth`
   - Xception: `models/xception_