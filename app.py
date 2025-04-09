import os
import streamlit as st
import numpy as np
from PIL import Image
import time
import cv2

from src.detector import DeepfakeDetector
from src.game import DeepfakeGame
from src.utils import (
    create_app_directories, 
    show_loading_spinner,
    display_prediction_with_explanation
)

# Set page configuration
st.set_page_config(
    page_title="DeepFake Detective",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for white theme and better UI
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 24px;
        background-color: #F5F5F5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6200EA;
        color: white;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 800;
        color: #6200EA;
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .real {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.5);
    }
    .fake {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.5);
    }
    .explainer {
        padding: 15px;
        background-color: #F5F5F5;
        border-radius: 5px;
        margin-top: 10px;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-container h1 {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """
    Load the deepfake detector model (cached)
    """
    create_app_directories()
    return DeepfakeDetector()

def analysis_mode():
    """
    Analysis mode UI for deepfake detection
    """
    st.markdown("""
    <div class="explainer">
        <h4>🔎 Analysis Mode</h4>
        <p>Upload an image containing faces to analyze it for potential deepfake manipulation. 
        The AI detector will process the image and provide a confidence score for each detected face.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    analyze_button = st.button("Analyze Image", type="primary")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process when the analyze button is clicked
        if analyze_button:
            detector = load_detector()
            
            # Show loading spinner
            show_loading_spinner()
            
            # Run detector on the uploaded image
            predictions, marked_image = detector.predict(image_data=uploaded_file)
            
            # Show results
            st.markdown("## Analysis Results")
            
            # Display predictions with explanations
            display_prediction_with_explanation(predictions, marked_image)
            
            # Technical explanation section
            with st.expander("How does the detection work?"):
                st.markdown("""
                The deepfake detector uses a state-of-the-art Xception neural network that has been trained on thousands of real and fake images. It works by:
                
                1. **Face Detection**: Locating all faces in the image
                2. **Feature Extraction**: Analyzing pixel patterns and inconsistencies
                3. **Classification**: Determining if the face is real or manipulated
                
                The model looks for subtle inconsistencies that are often invisible to the human eye, including:
                - Unnatural blending boundaries
                - Inconsistent texture patterns
                - Irregular noise distributions
                - Anomalies in facial features
                
                The confidence score represents how certain the model is about its prediction.
                """)

def game_mode():
    """
    Game mode UI for interactive deepfake spotting
    """
    # Initialize the game if not already in session state
    if 'game' not in st.session_state:
        detector = load_detector()
        st.session_state.game = DeepfakeGame(detector)
    
    game = st.session_state.game
    
    st.markdown("""
    <div class="explainer">
        <h4>🎮 Game Mode</h4>
        <p>Test your ability to spot deepfakes against our AI detector! 
        You'll be shown a series of images and need to guess whether each one is real or fake.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Game stats display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Score", value=game.score)
    with col2:
        st.metric(label="Questions", value=game.total_questions)
    with col3:
        accuracy = (game.score / game.total_questions * 100) if game.total_questions > 0 else 0
        st.metric(label="Accuracy", value=f"{accuracy:.1f}%")
    
    # Game controls
    if st.button("New Question", type="primary"):
        # Get a new question
        image_path, is_fake = game.get_new_question()
        
        if image_path:
            # Store the current question in session state
            st.session_state.current_image_path = image_path
            st.session_state.current_is_fake = is_fake
            st.session_state.answered = False
            
            # Display the image
            image = Image.open(image_path)
            st.session_state.image_display = st.image(image, caption="Is this image real or fake?", use_column_width=True)
            
            # Show the options
            st.session_state.options_col1, st.session_state.options_col2 = st.columns(2)
            with st.session_state.options_col1:
                st.session_state.real_button = st.button("Real", key="real_button", use_container_width=True)
            with st.session_state.options_col2:
                st.session_state.fake_button = st.button("Fake", key="fake_button", use_container_width=True)
        else:
            st.error("No sample images found. Please add images to the assets/sample_images directory.")
    
    # Handle user's answer
    if 'answered' in st.session_state and not st.session_state.answered:
        if st.session_state.real_button or st.session_state.fake_button:
            user_answer = st.session_state.fake_button  # True if Fake button is clicked
            
            # Evaluate the answer
            result = game.evaluate_answer(user_answer)
            
            if result:
                # Update answer status
                st.session_state.answered = True
                
                # Display the result
                if result['is_correct']:
                    st.success(f"Correct! You identified this as {'fake' if result['model_prediction'] else 'real'}.")
                else:
                    st.error(f"Wrong! This image is actually {'fake' if result['model_prediction'] else 'real'}.")
                
                # Show the marked image with model prediction
                st.image(result['marked_image'], caption=f"AI Detection Result - {'Fake' if result['model_prediction'] else 'Real'} ({result['model_confidence']*100:.1f}% confidence)", use_column_width=True)
                
                # Explanation
                if result['model_prediction']:  # If fake
                    st.info("The AI detected signs of manipulation in this image. Look for unnatural blending, inconsistent lighting, or irregular facial features.")
                else:  # If real
                    st.info("The AI determined this image is authentic. It has consistent texture patterns and natural facial features.")
    
    # Reset game button
    if st.button("Reset Game"):
        game.reset_game()
        st.experimental_rerun()
    
    # Instructions expander
    with st.expander("Game Instructions"):
        st.markdown("""
        ### How to Play
        1. Click **New Question** to get a random image
        2. Examine the image carefully
        3. Click **Real** if you think it's authentic or **Fake** if you think it's manipulated
        4. See the result and the AI's analysis
        5. Continue to test your skills against the AI!
        
        ### Tips for Spotting Deepfakes
        - Look for inconsistent lighting and shadows
        - Check for unnatural blending around facial features
        - Watch for irregular or blurry textures
        - Pay attention to the eyes and teeth, which are often poorly rendered
        - Notice any asymmetry in facial features
        """)

def main():
    """
    Main function to run the Streamlit app
    """
    # Header with logo
    st.markdown("""
    <div class="header-container">
        <h1>🕵️‍♀️ DeepFake Detective</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### AI-Powered Deepfake Detection System")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔎 Analysis Mode", "🎮 Game Mode", "ℹ️ About"])
    
    with tab1:
        analysis_mode()
        
    with tab2:
        game_mode()
        
    with tab3:
        st.markdown("""
        ## About DeepFake Detective
        
        DeepFake Detective is an advanced AI-powered system designed to detect manipulated facial images, commonly known as "deepfakes." Using state-of-the-art deep learning models, our system can identify subtle signs of manipulation that are often invisible to the human eye.
        
        ### Features
        
        - **High-Accuracy Detection**: Our model is trained on diverse deepfake techniques for reliable detection
        - **Detailed Analysis**: Get confidence scores and visual explanations of detection results
        - **Interactive Game Mode**: Test your ability to spot deepfakes against our AI
        
        ### How It Works
        
        DeepFake Detective uses a specialized Xception neural network architecture that has been trained on thousands of real and manipulated facial images. The model analyzes various aspects of the image, including:
        
        - Texture patterns and inconsistencies
        - Blending boundaries and artifacts
        - Noise distribution patterns
        - Facial feature coherence
        
        Our detector is specifically designed to identify artifacts introduced by various deepfake generation methods, including face-swapping, face reenactment, and GAN-generated faces.
        
        ### Educational Purpose
        
        This tool is created for educational and awareness purposes. As deepfake technology becomes more sophisticated and widespread, it's important for people to understand how to identify potentially manipulated content.
        
        ### Technologies Used
        
        - Python
        - PyTorch (Deep Learning)
        - OpenCV (Computer Vision)
        - Streamlit (Web Interface)
        - MTCNN (Face Detection)
        - Xception (Neural Network Architecture)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ | "
        "© 2025 DeepFake Detective | "
        "For educational purposes only"
    )

if __name__ == "__main__":
    main()