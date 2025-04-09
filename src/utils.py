import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import time

def load_sample_images(app_dir, num_samples=20):
    """
    Load sample images for the game mode, downloading them if necessary
    """
    real_dir = os.path.join(app_dir, 'assets/sample_images/real')
    fake_dir = os.path.join(app_dir, 'assets/sample_images/fake')
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Check if sample images already exist
    real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(real_images) >= num_samples and len(fake_images) >= num_samples:
        return True
    
    # TODO: Implement image download logic if needed
    # For now, we'll assume the user will provide sample images
    
    return False

def create_app_directories():
    """
    Create necessary directories for the application
    """
    os.makedirs('assets/sample_images/real', exist_ok=True)
    os.makedirs('assets/sample_images/fake', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def pil_to_numpy(pil_image):
    """
    Convert PIL Image to numpy array
    """
    return np.array(pil_image)

def numpy_to_pil(numpy_image):
    """
    Convert numpy array to PIL Image
    """
    return Image.fromarray(numpy_image.astype('uint8'))

def show_image_with_boxes(image, boxes, is_fakes, confidences):
    """
    Show image with bounding boxes and predictions
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for (x1, y1, x2, y2), is_fake, confidence in zip(boxes, is_fakes, confidences):
        color = 'red' if is_fake else 'green'
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                            fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        label = f"Fake: {confidence:.2f}" if is_fake else f"Real: {confidence:.2f}"
        ax.text(x1, y1 - 10, label, color=color, fontsize=12, 
                backgroundcolor='white')
    
    ax.axis('off')
    return fig

def display_prediction_with_explanation(predictions, image, width=None):
    """
    Display prediction results with explanation
    """
    if not predictions:
        st.warning("No faces detected in the image.")
        st.image(image, caption="Original Image", width=width)
        return
    
    st.image(image, caption="Analysis Result", width=width)
    
    for i, (box, is_fake, confidence) in enumerate(predictions):
        face_label = f"Face {i+1}" if len(predictions) > 1 else "Face"
        result_label = "FAKE" if is_fake else "REAL"
        confidence_pct = confidence * 100
        
        st.markdown(f"### {face_label}: {result_label} ({confidence_pct:.1f}% confidence)")
        
        # Colored progress bar
        bar_color = "rgba(255, 0, 0, 0.8)" if is_fake else "rgba(0, 255, 0, 0.8)"
        st.markdown(f"""
        <div style="width:100%; background-color:#ddd; height:30px; border-radius:5px;">
            <div style="width:{confidence_pct}%; background-color:{bar_color}; height:30px; border-radius:5px; text-align:center; line-height:30px; color:white;">
                {confidence_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation based on confidence
        if is_fake:
            if confidence > 0.95:
                explanation = "This image shows very strong signs of manipulation."
            elif confidence > 0.85:
                explanation = "This image likely contains significant manipulated elements."
            elif confidence > 0.7:
                explanation = "This image has several indicators of potential manipulation."
            elif confidence > 0.6:
                explanation = "This image has some subtle signs of manipulation."
            else:
                explanation = "This image has slight indications of manipulation, but the model is uncertain."
        else:
            if confidence > 0.95:
                explanation = "This image appears to be a genuine, unmanipulated photo."
            elif confidence > 0.85:
                explanation = "This image likely contains authentic, unmodified content."
            elif confidence > 0.7:
                explanation = "This image appears mostly authentic, with few suspicious elements."
            elif confidence > 0.6:
                explanation = "This image has some authentic qualities, but contains ambiguous elements."
            else:
                explanation = "The model is having difficulty determining if this is authentic."
                
        st.info(explanation)

def show_loading_spinner():
    """
    Show a loading spinner with custom message
    """
    with st.spinner("Analyzing image... This may take a few seconds."):
        # Simulate processing time for better UX
        time.sleep(0.5)
        placeholder = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            
            if i == 20:
                placeholder.info("Detecting faces...")
            elif i == 40:
                placeholder.info("Extracting features...")
            elif i == 60:
                placeholder.info("Running deepfake analysis...")
            elif i == 80:
                placeholder.info("Generating results...")
        
        placeholder.empty()
        progress_bar.empty()