import os
import streamlit as st
import numpy as np
from PIL import Image
import time
import cv2

# Import the standard detector and the alternate approach
from src.detector import DeepfakeDetector
from src.game import DeepfakeGame
from src.utils import (
    create_app_directories, 
    show_loading_spinner,
    display_prediction_with_explanation
)

# Add this at the top of your app.py file, right after the imports:
try:
    # Try to import the alternate detector (only if needed)
    from src.alternate_model_approach import create_efficientnet_detector
    HAS_ALTERNATE_MODEL = True
except ImportError:
    HAS_ALTERNATE_MODEL = False

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
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
    }
    .image-item {
        border: 2px solid transparent;
        border-radius: 8px;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .image-item:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .image-item.selected {
        border-color: #6200EA;
        box-shadow: 0 0 12px rgba(98, 0, 234, 0.5);
    }
    .image-container {
        position: relative;
    }
    .image-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 5px;
        color: white;
        font-size: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """
    Load the deepfake detector model (cached)
    Try multiple approaches if needed
    """
    create_app_directories()
    
    # First try the standard approach
    detector = DeepfakeDetector()
    
    # Check if we're getting 0.5 confidence consistently
    if detector.model_type == "dummy" and HAS_ALTERNATE_MODEL:
        # Fall back to EfficientNet if Xception fails
        print("Xception model failed to load. Trying EfficientNet instead.")
        detector = create_efficientnet_detector()
    
    return detector

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
    
    # Image size control
    col1, col2 = st.columns([3, 1])
    with col2:
        img_size = st.select_slider(
            "Image Display Size", 
            options=["Small", "Medium", "Large"],
            value="Medium"
        )
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    analyze_button = st.button("Analyze Image", type="primary")
    
    if uploaded_file is not None:
        # Display the uploaded image with appropriate size
        image = Image.open(uploaded_file)
        
        # Get image display width based on selected size
        if img_size == "Small":
            width = 300
        elif img_size == "Medium":
            width = 500
        else:
            width = 800
            
        st.image(image, caption="Uploaded Image", width=width)
        
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
            display_prediction_with_explanation(predictions, marked_image, width)
            
            # Technical explanation section
            with st.expander("Technical Details"):
                st.markdown("""
                The model analyzes facial features, texture patterns, and other visual cues to determine if an image has been manipulated.
                """)
            
            # Game instructions
            with st.expander("Game Instructions"):
                st.markdown("""
                ### How to Play
                
                1. **Any images will work!** You've already added some random real and fake face images which is perfect.
                2. Simply click on any image thumbnail in the grid to select it
                3. After selecting an image, choose whether you think it's **REAL** or **FAKE**
                4. See if your guess matches the AI detection
                5. Continue selecting different images to test your skills!
                
                ### About Your Sample Images
                
                - The app uses whatever images you've placed in the sample folders
                - Real images should be in `assets/sample_images/real/`
                - Fake images should be in `assets/sample_images/fake/`
                - You can add more images to these folders anytime
                
                ### Tips for Spotting Deepfakes
                
                - Look for inconsistent lighting and shadows
                - Check for unnatural blending around facial features
                - Watch for irregular or blurry textures
                - Pay attention to the eyes and teeth, which are often poorly rendered
                - Notice any asymmetry in facial features
                """)

def game_mode():
    """
    Game mode UI for interactive deepfake spotting with grid display
    """
    # Initialize the game if not already in session state
    if 'game' not in st.session_state:
        detector = load_detector()
        st.session_state.game = DeepfakeGame(detector)
    
    game = st.session_state.game
    
    # Initialize session state variables for game
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    
    if 'answer_submitted' not in st.session_state:
        st.session_state.answer_submitted = False
    
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    st.markdown("""
    <div class="explainer">
        <h4>🎮 Game Mode</h4>
        <p>Test your ability to spot deepfakes against our AI detector! 
        Click on any image below, then determine if it's real or fake.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all available images
    all_images, image_labels = game.get_all_images()
    
    if not all_images:
        st.warning("No sample images found. Please add images to the assets/sample_images directory.")
        return
    
    # Game stats display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Score", value=game.score)
    with col2:
        st.metric(label="Questions", value=game.total_questions)
    with col3:
        accuracy = (game.score / game.total_questions * 100) if game.total_questions > 0 else 0
        st.metric(label="Accuracy", value=f"{accuracy:.1f}%")
    
    # Image selection grid
    st.subheader("Select an image to analyze:")
    
    # Image size control
    img_display_size = st.select_slider(
        "Thumbnail Size", 
        options=["Small", "Medium", "Large"],
        value="Medium"
    )
    
    # Set number of columns based on thumbnail size
    if img_display_size == "Small":
        num_cols = 6
        thumb_size = 100
    elif img_display_size == "Medium":
        num_cols = 4
        thumb_size = 150
    else:
        num_cols = 3
        thumb_size = 200
    
    # Create image grid
    cols = st.columns(num_cols)
    for i, img_path in enumerate(all_images):
        # Determine which column to place this image
        col_idx = i % num_cols
        
        with cols[col_idx]:
            # Load and resize image
            img = Image.open(img_path)
            img.thumbnail((thumb_size, thumb_size))
            
            # Generate a unique key for this image
            img_key = f"img_{i}"
            
            # Determine if this image is selected
            is_selected = st.session_state.selected_image == img_path
            
            # Add CSS class for selected image
            container_class = "image-item selected" if is_selected else "image-item"
            
            # Create a container with the appropriate class
            st.markdown(f'<div class="{container_class}" id="{img_key}">', unsafe_allow_html=True)
            
            # Display the image
            if st.image(img, use_column_width=True, output_format="PNG"):
                # This won't actually execute - we'll use a separate button
                pass
            
            # Close the container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a button with the same key to capture clicks
            if st.button(f"Select", key=img_key):
                st.session_state.selected_image = img_path
                st.session_state.answer_submitted = False
                # Force a rerun to update the UI
                st.experimental_rerun()
    
    # Display the selected image and controls
    if st.session_state.selected_image:
        st.markdown("---")
        st.subheader("Your selected image:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the selected image
            img = Image.open(st.session_state.selected_image)
            
            # Adjust display size
            if img_display_size == "Small":
                width = 300
            elif img_display_size == "Medium":
                width = 400
            else:
                width = 600
                
            st.image(img, width=width)
        
        with col2:
            st.markdown("### Is this image real or fake?")
            real_col, fake_col = st.columns(2)
            
            with real_col:
                real_button = st.button("REAL", use_container_width=True, 
                                       type="primary" if not st.session_state.answer_submitted else "secondary")
            
            with fake_col:
                fake_button = st.button("FAKE", use_container_width=True,
                                       type="primary" if not st.session_state.answer_submitted else "secondary")
            
            # Handle user's answer
            if (real_button or fake_button) and not st.session_state.answer_submitted:
                user_answer = fake_button  # True if Fake button is clicked, False if Real
                
                # Evaluate the answer
                result = game.evaluate_answer(st.session_state.selected_image, user_answer)
                
                if result:
                    # Save result and mark as submitted
                    st.session_state.last_result = result
                    st.session_state.answer_submitted = True
                    
                    # Force a rerun to update the UI
                    st.experimental_rerun()
        
        # Display result after submission
        if st.session_state.answer_submitted and st.session_state.last_result:
            result = st.session_state.last_result
            
            st.markdown("---")
            st.subheader("Analysis Result:")
            
            # Display success or error message
            if result['is_correct']:
                st.success(f"Correct! This image is {'fake' if result['actual_is_fake'] else 'real'}.")
            else:
                st.error(f"Wrong! This image is actually {'fake' if result['actual_is_fake'] else 'real'}.")
            
            # Show the marked image with model prediction
            marked_image = result['marked_image']
            
            # Adjust display size
            if img_display_size == "Small":
                width = 400
            elif img_display_size == "Medium":
                width = 500
            else:
                width = 700
                
            st.image(marked_image, 
                    caption=f"AI Detection Result - {'Fake' if result['model_prediction'] else 'Real'} "
                           f"({result['model_confidence']*100:.1f}% confidence)", 
                    width=width)
            
            # Explanation
            if result['actual_is_fake']:  # If fake
                st.info("This is a manipulated image. Look for unnatural blending, inconsistent lighting, or irregular facial features.")
            else:  # If real
                st.info("This is an authentic image. It has consistent texture patterns and natural facial features.")
    
    # Reset game button
    if st.button("Reset Game"):
        game.reset_game()
        st.session_state.selected_image = None
        st.session_state.answer_submitted = False
        st.session_state.last_result = None
        st.experimental_rerun()
    
    # Instructions expander
    with st.expander("Game Instructions"):
        st.markdown("""
        ### How to Play
        1. Click on any image from the grid to select it
        2. Examine the image carefully
        3. Click **REAL** if you think it's authentic or **FAKE** if you think it's manipulated
        4. See the result and the AI's analysis
        5. Continue with another image to test your skills!
        
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