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

    # Simple blue / white website style
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    header,
    #MainMenu,
    .reportview-container .main footer {
        visibility: hidden;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background: #F5FAFF;
    }

    .stApp {
        background: #F5FAFF;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
    }

    .page-wrapper {
        max-width: 850px;
        margin: 0 auto;
        padding: 30px 20px 40px;
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 28px 20px;
        background: white;
        border: 1px solid #D7E7FF;
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(0, 82, 204, 0.08);
        margin-bottom: 24px;
    }

    .hero-logo {
        width: 90px;
        height: 90px;
        border-radius: 22px;
        background: linear-gradient(135deg, #E6F0FF 0%, #CDE5FF 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #C6DBFF;
    }

    .hero-title {
        font-size: 2.1rem;
        color: #003A8C;
        font-weight: 800;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #1D4ED8;
        font-size: 1rem;
        margin-top: 8px;
        max-width: 620px;
    }

    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }

    .feature-card {
        background: white;
        border: 1px solid #D7E7FF;
        border-radius: 16px;
        padding: 18px;
        text-align: left;
    }

    .feature-title {
        color: #003A8C;
        font-weight: 700;
        margin-top: 10px;
        font-size: 0.96rem;
    }

    .feature-text {
        color: #475569;
        margin-top: 8px;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .section-card {
        background: white;
        border: 1px solid #D7E7FF;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #003A8C;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .input-label {
        color: #003A8C;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-bottom: 12px;
        display: block;
    }

    .stButton > button {
        background: #0052CC !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }

    .stButton > button:hover {
        background: #003A8C !important;
    }

    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        color: #003A8C !important;
        border: 1px solid #BDD7FF !important;
        border-radius: 10px !important;
    }

    .stFileUploader > div {
        background: #FFFFFF !important;
        border: 1px solid #BDD7FF !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }

    .result-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 22px;
        align-items: start;
    }

    .image-card,
    .result-card {
        background: white;
        border: 1px solid #D7E7FF;
        border-radius: 18px;
        padding: 22px;
    }

    .result-box {
        border: 1px solid #0052CC;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        background: #EFF6FF;
    }

    .result-label {
        color: #003A8C;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 14px;
    }

    .confidence-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 18px;
    }

    .confidence-value {
        color: #003A8C;
        font-size: 2.4rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .icon-large {
        font-size: 3rem;
    }

    .small-note {
        color: #64748B;
        font-size: 0.87rem;
    }

    @media (max-width: 760px) {
        .result-row,
        .feature-row {
            grid-template-columns: 1fr;
        }

        .hero {
            padding: 20px 0;
        }
    }

    .stAppFooter,
    .reportview-container .main footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

    col_logo, col_text = st.columns([1, 3], gap='small')
    with col_logo:
        st.image(create_logo(), width=90)
    with col_text:
        st.markdown('''
            <div class="hero">
                <div class="hero-title">Eye State Detector</div>
                <div class="hero-subtitle">Upload an eye image and get fast, clean open/closed prediction with confidence.</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="feature-row">', unsafe_allow_html=True)
    st.markdown('''
        <div class="feature-card">
            <div class="feature-title">Eye classifier</div>
            <div class="feature-text">Smart model that detects whether eyes are open or closed in a single image.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Instant results</div>
            <div class="feature-text">Prediction appears immediately after upload, with a clear confidence score.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Modern UI</div>
            <div class="feature-text">Minimal, professional website style with clean white and blue design.</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    model_path = 'eye_model.h5'
    if not os.path.exists(model_path):
        st.error(f'Model file not found: {model_path}')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    classifier = load_classifier(model_path)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload or capture your image</div>', unsafe_allow_html=True)
    st.markdown('<label class="input-label">Select image source</label>', unsafe_allow_html=True)

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

    image = None

    if st.session_state.input_mode == 'upload':
        uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed', key='file_upload')
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

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction display
    if image is not None:
        st.markdown('<div class="result-row">', unsafe_allow_html=True)

        with st.container():
            col_img, col_result = st.columns([1, 1], gap='large')

            with col_img:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(image, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_result:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                with st.spinner('Analyzing...'):
                    label, confidence = predict_image(classifier, image)

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if label == 'Eyes Open':
                    st.markdown('<div class="icon-large">👁️</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-label">Eyes Open</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="icon-large">😴</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-label">Eyes Closed</div>', unsafe_allow_html=True)

                st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-value">{confidence:.1%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
