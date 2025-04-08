import os
import streamlit as st
import time
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path

# Import local modules
import config
from utils.detector import DeepfakeDetector
from utils.game import DeepfakeGame

# Set page configuration
st.set_page_config(
    page_title="DeepFake Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open(os.path.join("assets", "style.css")) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = DeepfakeDetector()
if 'game' not in st.session_state:
    st.session_state.game = DeepfakeGame(detector=st.session_state.detector)
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'game_result' not in st.session_state:
    st.session_state.game_result = None
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

# Function to start a new game
def start_new_game():
    st.session_state.current_question = st.session_state.game.new_game()
    st.session_state.game_result = None
    st.session_state.game_over = False

# Function to handle user answer in game mode
def handle_answer(answer):
    result = st.session_state.game.submit_answer(answer)
    st.session_state.game_result = result
    
    if result.get("game_complete", False):
        st.session_state.game_over = True
    else:
        st.session_state.current_question = result.get("next_question")

# Main app header
st.markdown("<h1 class='main-header'>DeepFake Detection Platform</h1>", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a mode:", ["Detection Mode", "Game Mode", "About"])

# Mode-specific content
if app_mode == "Detection Mode":
    st.markdown("<h2 class='sub-header'>Deepfake Detection</h2>", unsafe_allow_html=True)
    
    # Information box
    st.markdown("""
    <div class='info-box'>
        <p><strong>Upload an image</strong> to detect if it contains deepfake faces. 
        Our AI model will analyze the image and provide a prediction with confidence score.</p>
        <p>The model is trained to detect facial manipulations in images, including face swaps, 
        GAN-generated faces, and facial reenactments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Progress bar for analysis
        with col2:
            with st.spinner("Analyzing image..."):
                # Process the image
                result = st.session_state.detector.predict(image, return_visualization=True)
                
                # Display result
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    # Display result with appropriate styling
                    result_class = "real-result" if prediction == "real" else "fake-result"
                    st.markdown(f"""
                    <div class='detection-result {result_class}'>
                        <h3>Detection Result: {prediction.upper()}</h3>
                        <p>Confidence: {confidence:.2f}</p>
                        <div class='confidence-bar' style='background: linear-gradient(to right, {'#10B981' if prediction == 'real' else '#EF4444'} {int(confidence * 100)}%, #E5E7EB {int(confidence * 100)}%);'></div>
                        <p class='confidence-text'>{'Real: ' if prediction == 'real' else 'Fake: '}{confidence:.2f}</p>
                        <p class='confidence-text'>{'Fake: ' if prediction == 'real' else 'Real: '}{1 - confidence:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display visualization
                    if "visualization" in result:
                        st.image(result["visualization"], caption="Detection Visualization", use_column_width=True)
    
    # Sample images section
    st.markdown("<h3 class='sub-header'>Try with sample images</h3>", unsafe_allow_html=True)
    
    # Create sample images directory if it doesn't exist
    sample_dir = config.SAMPLE_IMAGES_DIR
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check if sample images exist, if not, create some
    sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(sample_images) < 4:
        # Create sample real and fake images
        for i in range(2):
            # Sample real image (green-tinted)
            img = Image.new('RGB', (300, 300), color=(100, 200, 100))
            img.save(os.path.join(sample_dir, f"sample_real_{i}.jpg"))
            
            # Sample fake image (red-tinted)
            img = Image.new('RGB', (300, 300), color=(200, 100, 100))
            img.save(os.path.join(sample_dir, f"sample_fake_{i}.jpg"))
        
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Display sample images in a grid
    sample_cols = st.columns(4)
    for i, sample_file in enumerate(sample_images[:4]):
        with sample_cols[i % 4]:
            sample_path = os.path.join(sample_dir, sample_file)
            sample_img = Image.open(sample_path)
            
            # Add a caption based on filename
            is_fake = "fake" in sample_file.lower()
            caption = "Fake Image" if is_fake else "Real Image"
            
            # Create a clickable image
            st.image(sample_img, caption=caption, use_column_width=True)
            if st.button(f"Analyze {caption}", key=f"sample_{i}"):
                # Set the sample image as the current image
                st.session_state.current_sample = sample_path
                st.experimental_rerun()
    
    # Process selected sample image
    if hasattr(st.session_state, 'current_sample'):
        sample_path = st.session_state.current_sample
        sample_img = Image.open(sample_path)
        
        st.markdown("<h3 class='sub-header'>Sample Image Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(sample_img, caption="Sample Image", use_column_width=True)
        
        with col2:
            with st.spinner("Analyzing sample image..."):
                # Process the image
                result = st.session_state.detector.predict(sample_img, return_visualization=True)
                
                # Display result
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    # Display result with appropriate styling
                    result_class = "real-result" if prediction == "real" else "fake-result"
                    st.markdown(f"""
                    <div class='detection-result {result_class}'>
                        <h3>Detection Result: {prediction.upper()}</h3>
                        <p>Confidence: {confidence:.2f}</p>
                        <div class='confidence-bar' style='background: linear-gradient(to right, {'#10B981' if prediction == 'real' else '#EF4444'} {int(confidence * 100)}%, #E5E7EB {int(confidence * 100)}%);'></div>
                        <p class='confidence-text'>{'Real: ' if prediction == 'real' else 'Fake: '}{confidence:.2f}</p>
                        <p class='confidence-text'>{'Fake: ' if prediction == 'real' else 'Real: '}{1 - confidence:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display visualization
                    if "visualization" in result:
                        st.image(result["visualization"], caption="Detection Visualization", use_column_width=True)

# Game Mode
elif app_mode == "Game Mode":
    st.markdown("<h2 class='sub-header'>Deepfake Detection Game</h2>", unsafe_allow_html=True)
    
    # Information box
    st.markdown("""
    <div class='info-box'>
        <p><strong>Test your skills against AI</strong> in detecting deepfakes! You'll be shown a series of images, 
        and you need to decide if each one is real or fake.</p>
        <p>After each answer, you'll see how you did compared to our AI detection model. Can you beat the machine?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Start a new game button
    if st.button("Start New Game") or st.session_state.current_question is None:
        start_new_game()
    
    # Game container
    st.markdown("<div class='game-container'>", unsafe_allow_html=True)
    
    # Display current scores if game is in progress
    if st.session_state.game_result is not None:
        st.markdown(f"""
        <div class='score-container'>
            <div class='score-box user-score'>
                <h3>Your Score</h3>
                <p>{st.session_state.game_result["user_score"]} / {st.session_state.game_result["question_number"]}</p>
            </div>
            <div class='score-box ai-score'>
                <h3>AI Score</h3>
                <p>{st.session_state.game_result["ai_score"]} / {st.session_state.game_result["question_number"]}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display current question
    if st.session_state.current_question and not st.session_state.game_over:
        st.markdown(f"""
        <h3 class='sub-header'>Question {st.session_state.current_question["question_number"]} of {st.session_state.current_question["total_questions"]}</h3>
        """, unsafe_allow_html=True)
        
        # Display image
        image_path = st.session_state.current_question["image_path"]
        image = Image.open(image_path)
        
        st.markdown("<div class='question-container'>", unsafe_allow_html=True)
        st.image(image, caption="Is this image real or fake?", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Answer buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("REAL", key="real_button", use_container_width=True):
                handle_answer("real")
        with col2:
            if st.button("FAKE", key="fake_button", use_container_width=True):
                handle_answer("fake")
    
    # Display result of previous question
    if st.session_state.game_result is not None and not st.session_state.game_over:
        is_correct = st.session_state.game_result["is_correct"]
        result_class = "correct-result" if is_correct else "incorrect-result"
        
        st.markdown(f"""
        <div class='result-container {result_class}'>
            <h3>{"Correct! ✓" if is_correct else "Incorrect! ✗"}</h3>
            <p>The image was <strong>{st.session_state.game_result["true_label"].upper()}</strong></p>
            <p>AI predicted: <strong>{st.session_state.game_result["ai_prediction"].upper()}</strong> 
               (Confidence: {st.session_state.game_result["ai_confidence"]:.2f})</p>
            <p>AI was <strong>{"correct" if st.session_state.game_result["ai_is_correct"] else "wrong"}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Game over screen
    if st.session_state.game_over:
        st.markdown("<h3 class='sub-header'>Game Over!</h3>", unsafe_allow_html=True)
        
        # Display final scores
        final_user_score = st.session_state.game_result["final_user_score"]
        final_ai_score = st.session_state.game_result["final_ai_score"]
        total_questions = st.session_state.game_result["total_questions"]
        winner = st.session_state.game_result["winner"]
        
        if winner == "user":
            result_message = "Congratulations! You beat the AI! 🎉"
            result_class = "success-box"
        elif winner == "ai":
            result_message = "The AI won this round. Better luck next time! 🤖"
            result_class = "warning-box"
        else:
            result_message = "It's a tie! You matched the AI's performance. 🤝"
            result_class = "info-box"
        
        st.markdown(f"""
        <div class='{result_class}'>
            <h3>{result_message}</h3>
        </div>
        
        <div class='score-container'>
            <div class='score-box user-score'>
                <h3>Your Final Score</h3>
                <p>{final_user_score} / {total_questions}</p>
                <p>({(final_user_score / total_questions * 100):.1f}%)</p>
            </div>
            <div class='score-box ai-score'>
                <h3>AI Final Score</h3>
                <p>{final_ai_score} / {total_questions}</p>
                <p>({(final_ai_score / total_questions * 100):.1f}%)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # New game button
        if st.button("Play Again"):
            start_new_game()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Game statistics
    if hasattr(st.session_state, 'game') and st.session_state.game:
        game_stats = st.session_state.game.get_game_stats()
        
        if game_stats["total_games"] > 0:
            st.markdown("<h3 class='sub-header'>Your Game Statistics</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Games Played", game_stats["total_games"])
            with col2:
                st.metric("Your Wins", game_stats["user_wins"])
            with col3:
                st.metric("AI Wins", game_stats["ai_wins"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Average Score", f"{game_stats['avg_user_score']:.1f}")
            with col2:
                st.metric("AI Average Score", f"{game_stats['avg_ai_score']:.1f}")

# About page
elif app_mode == "About":
    st.markdown("<h2 class='sub-header'>About This Platform</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>This DeepFake Detection Platform is designed to help users identify manipulated media. 
        With the rise of AI-generated content, being able to distinguish between real and fake media 
        is becoming increasingly important.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>How It Works</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Our platform uses deep learning models trained on datasets of both real and manipulated faces to detect deepfakes.
    The system looks for subtle inconsistencies and artifacts that are often present in manipulated media.
    
    The detection process involves:
    1. **Face Detection** - Identifying and extracting faces from the uploaded image
    2. **Feature Extraction** - Analyzing facial features and image characteristics
    3. **Classification** - Determining whether each face is real or manipulated
    4. **Confidence Scoring** - Providing a probability estimate of the prediction
    """)
    
    st.markdown("<h3 class='sub-header'>Technology Stack</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='tech-container'>
        <div class='tech-box'>
            <h4>Frontend</h4>
            <ul>
                <li>Streamlit</li>
                <li>Matplotlib</li>
                <li>PIL (Python Imaging Library)</li>
            </ul>
        </div>
        <div class='tech-box'>
            <h4>Backend & ML</h4>
            <ul>
                <li>PyTorch</li>
                <li>EfficientNet / Xception</li>
                <li>MTCNN (Face Detection)</li>
            </ul>
        </div>
        <div class='tech-box'>
            <h4>Training Data</h4>
            <ul>
                <li>FaceForensics++</li>
                <li>Celeb-DF</li>
                <li>DFDC</li>
            </ul>
        </div>
        <div class='tech-box'>
            <h4>Deployment</h4>
            <ul>
                <li>Python 3.8+</li>
                <li>Docker (optional)</li>
                <li>Streamlit Cloud</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>Limitations</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
        <p>While our system is designed to be accurate, it has some limitations:</p>
        <ul>
            <li>No detection system is 100% accurate - there will be false positives and false negatives</li>
            <li>The technology is in a constant race with deepfake creation methods</li>
            <li>Very high-quality deepfakes can sometimes fool detection systems</li>
            <li>Low-quality images or those with obstructed faces may result in less accurate predictions</li>
        </ul>
        <p>Remember that this tool should be used as one element in a critical evaluation of media authenticity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2025 DeepFake Detection Platform</div>", unsafe_allow_html=True)

# Initialize empty files directory in utils module if not present
if not os.path.exists(os.path.join("utils", "__init__.py")):
    os.makedirs("utils", exist_ok=True)
    with open(os.path.join("utils", "__init__.py"), "w") as f:
        f.write("# Init file for utils module\n")

# Main app entry point
if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass