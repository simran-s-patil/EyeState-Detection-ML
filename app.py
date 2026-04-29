import os
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import requests

from eye_classifier import EyeClassifier


def create_logo():
    """Create a simple professional eye icon"""
    img = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(16, 185, 129), outline=(5, 150, 100), width=3)
    draw.ellipse([45, 40, 55, 55], fill=(31, 41, 55))
    draw.ellipse([120, 20, 180, 80], fill=(16, 185, 129), outline=(5, 150, 100), width=3)
    draw.ellipse([145, 40, 155, 55], fill=(31, 41, 55))
    return img


@st.cache_resource
def load_classifier(model_path: str = 'eye_model.h5') -> EyeClassifier:
    classifier = EyeClassifier()
    classifier.load_model(model_path)
    return classifier


def predict_image(classifier: EyeClassifier, image: Image.Image):
    image = image.convert('L')
    image = image.resize(classifier.img_size)
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=[0, -1])
    prediction = classifier.model.predict(image_array, verbose=0)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = 'Eyes Open' if prediction > 0.5 else 'Eyes Closed'
    return label, confidence


def main():
    st.set_page_config(
        page_title='Eye State Detector',
        page_icon='👁️',
        layout='centered',
        initial_sidebar_state='collapsed',
    )

    # Blue & White Simple Theme CSS
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background: #FFFFFF;
    }
    
    .stApp {
        background: #FFFFFF;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
    }
    
    .header-container {
        background: #0052CC;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 82, 204, 0.15);
    }
    
    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.9rem;
        margin-top: 6px;
        font-weight: 400;
    }
    
    .main-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 30px 20px;
    }
    
    .input-label {
        color: #0052CC;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        display: block;
    }
    
    .button-row {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        justify-content: center;
    }
    
    .prediction-area {
        display: flex;
        flex-direction: column;
        gap: 20px;
        align-items: center;
    }
    
    .image-section {
        background: #F5F5F5;
        border: 2px solid #0052CC;
        border-radius: 12px;
        padding: 20px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 300px;
    }
    
    .result-box-open {
        background: #0052CC;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        width: 100%;
    }
    
    .result-box-closed {
        background: #0052CC;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        width: 100%;
    }
    
    .result-label {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .result-icon {
        font-size: 3rem;
        margin-bottom: 10px;
        display: block;
    }
    
    .confidence-label {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 12px;
    }
    
    .confidence-value {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 6px;
    }
    
    .stButton > button {
        background: #0052CC !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        background: #0041A3 !important;
    }
    
    .stFileUploader {
        background: transparent !important;
    }
    
    .stTextInput > div > div > input {
        background: white !important;
        color: #0052CC !important;
        border: 1px solid #0052CC !important;
    }
    
    .stCameraInput > div > div {
        background: transparent !important;
    }
    
    [data-testid="stCameraInput"] button {
        background: #0052CC !important;
        color: white !important;
        border: none !important;
    }
    
    .stSpinner > div > div {
        border-color: #0052CC !important;
    }
    
    .stSpinner > div > div > div {
        background-color: #0052CC !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">👁️ Eye State Detector</div>
        <div class="header-subtitle">Analyze Eye Status</div>
    </div>
    """, unsafe_allow_html=True)

    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    model_path = 'eye_model.h5'
    if not os.path.exists(model_path):
        st.error(f'Model file not found: {model_path}')
        return

    classifier = load_classifier(model_path)

    # Input selection
    st.markdown('<label class="input-label">Select Image Source</label>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap='small')
    with col1:
        upload_tab = st.button('📁 Upload', use_container_width=True, key='upload_btn')
    with col2:
        url_tab = st.button('🔗 URL', use_container_width=True, key='url_btn')
    with col3:
        camera_tab = st.button('📷 Camera', use_container_width=True, key='camera_btn')

    if upload_tab:
        st.session_state.input_mode = 'upload'
    elif url_tab:
        st.session_state.input_mode = 'url'
    elif camera_tab:
        st.session_state.input_mode = 'camera'

    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = 'upload'

    st.markdown('<br>', unsafe_allow_html=True)

    image = None

    if st.session_state.input_mode == 'upload':
        uploaded_file = st.file_uploader('Choose image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed', key='file_upload')
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif st.session_state.input_mode == 'url':
        image_url = st.text_input('', placeholder='Paste image URL', label_visibility='collapsed', key='url_input')
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f'Error: {e}')

    elif st.session_state.input_mode == 'camera':
        camera_image = st.camera_input('', label_visibility='collapsed', key='camera_input')
        if camera_image:
            image = Image.open(camera_image)

    # Prediction display
    if image is not None:
        st.markdown('<div class="prediction-area">', unsafe_allow_html=True)

        # Image display
        st.markdown('<div class="image-section">', unsafe_allow_html=True)
        st.image(image, width=500)
        st.markdown('</div>', unsafe_allow_html=True)

        # Result display
        st.markdown('<br>', unsafe_allow_html=True)

        with st.spinner('Analyzing...'):
            label, confidence = predict_image(classifier, image)

        if label == 'Eyes Open':
            st.markdown(f"""
            <div class="result-box-open">
                <span class="result-icon">👁️</span>
                <div class="result-label">Eyes Open</div>
                <div class="confidence-label">Confidence</div>
                <div class="confidence-value">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-closed">
                <span class="result-icon">😴</span>
                <div class="result-label">Eyes Closed</div>
                <div class="confidence-label">Confidence</div>
                <div class="confidence-value">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
