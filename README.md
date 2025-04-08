# DeepFake Detection Platform

An interactive Streamlit application for detecting deepfake images and testing your ability to spot AI-generated faces.

## Features

- **Detection Mode**: Upload images and get AI-powered predictions on whether they contain deepfake faces
- **Game Mode**: Test your deepfake detection skills against AI in an interactive quiz game
- **Pre-trained Models**: Uses state-of-the-art EfficientNet and Xception architectures
- **Face Detection**: Automatically extracts and analyzes faces in images
- **Visualization**: See highlighted regions of potential manipulation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone this repository or download the source code:

```bash
git clone https://github.com/yourusername/deepfake-detection-app.git
cd deepfake-detection-app
```

2. Create a virtual environment (optional but recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Downloading Pre-trained Models (Optional)

The application will work with the pre-trained EfficientNet and Xception models from torchvision/timm. However, for better deepfake detection accuracy, you can download fine-tuned models:

1. Create a `models` directory if it doesn't exist:

```bash
mkdir -p models
```

2. Download the fine-tuned models:

- For EfficientNet: [Download Link](https://example.com/efficientnet_deepfake_model.pth)
- For Xception: [Download Link](https://example.com/xception_deepfake_model.pth)

3. Place the downloaded models in the `models` directory:

```
deepfake-detection-app/
└── models/
    ├── efficientnet_model.pth
    └── xception_model.pth
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

This will launch the application in your default web browser, typically at `http://localhost:8501`.

## Using the Application

### Detection Mode

1. Upload an image using the file uploader
2. The application will process the image and display the detection results
3. Results include:
   - Classification (Real or Fake)
   - Confidence score
   - Visualization highlighting potential manipulations

### Game Mode

1. Click "Start New Game" to begin
2. For each image shown, click either "REAL" or "FAKE"
3. After each answer, see if you were correct and how you compared to the AI
4. Complete the game to see your final score and statistics

## Project Structure

```
deepfake-detection-app/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── models/                 # Directory for pre-trained models
├── utils/
│   ├── __init__.py
│   ├── detector.py         # Deepfake detection functionality
│   └── game.py             # Game mode functionality 
├── data/
│   ├── game_images/        # Images for the game mode
│   │   ├── real/           # Real images for the game
│   │   └── fake/           # Fake/deepfake images for the game
│   └── sample_images/      # Sample images for demo purposes
└── assets/
    └── style.css           # Custom styling for the app
```

## Customization

### Adding Your Own Game Images

To add your own images to the game:

1. Place real images in the `data/game_images/real/` directory
2. Place fake/deepfake images in the `data/game_images/fake/` directory

The images should be in JPG, JPEG, or PNG format.

### Changing the Model

You can switch between EfficientNet and Xception models by modifying the `DEFAULT_MODEL` parameter in `config.py`:

```python
# Use either 'efficientnet' or 'xception'
DEFAULT_MODEL = 'efficientnet'
```

## Limitations

- The accuracy of deepfake detection depends on the quality of the input images
- Very high-quality deepfakes may sometimes evade detection
- Low-resolution or heavily compressed images may yield less reliable results
- The model is primarily designed for facial deepfake detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The models are trained using datasets from [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), and [DFDC](https://ai.facebook.com/datasets/dfdc/)
- Face detection is performed using [MTCNN](https://github.com/timesler/facenet-pytorch)
- Deep learning models built with [PyTorch](https://pytorch.org/) and [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)