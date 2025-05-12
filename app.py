import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import gdown
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Cache the model loading for better performance
@st.cache_resource
def load_my_model():
    try:
        model_url = 'https://drive.google.com/uc?id=1_vd-edYRk2A7tVFg-E951xL2oqBePmQM'  # Replace with your actual file ID
        model_path = 'food_freshness_model.h5'

        # Download model from Google Drive
        if not os.path.exists(model_path):
            gdown.download(model_url, model_path, quiet=False)

        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_my_model()

# Class names
class_names = ['freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber', 'freshokra',
               'freshoranges', 'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottenbittergroud',
               'rottencapsicum', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato']

# Helper function to process prediction
def get_prediction(img_path):
    try:
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Split the class name
        if predicted_class.startswith("fresh"):
            status = "Fresh 🟢"
            fruit = predicted_class.replace("fresh", "").capitalize()
        else:
            status = "Rotten 🔴"
            fruit = predicted_class.replace("rotten", "").capitalize()

        confidence = np.max(prediction) * 100
        
        # Fix some fruit names
        if fruit.lower() == "bittergroud":
            fruit = "Bitter Gourd"
        elif fruit.lower() == "okra":
            fruit = "Okra (Lady Finger)"
            
        return fruit, status, confidence
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

# Custom CSS for better appearance
st.markdown("""
    <style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #2e86ab !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #2e86ab;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f8f9fa;
    }
    .fresh {
        color: #28a745;
        font-weight: bold;
    }
    .rotten {
        color: #dc3545;
        font-weight: bold;
    }
    .confidence {
        font-size: 14px;
        color: #6c757d;
    }
    .footer {
        margin-top: 50px;
        font-size: 12px;
        text-align: center;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

# Main app
st.markdown('<div class="header">RotBot: AI Visual Classifier for Food Spoilage</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
    Upload an image of a fruit or vegetable to check if it's fresh or rotten.
    <br>Supported items: Apples, Banana, Bitter Gourd, Capsicum, Cucumber, Okra, Oranges, Potato, Tomato
    </div>
    """, unsafe_allow_html=True)

# Image upload
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], key="uploader", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', width=200)
    
    # Process image
    with st.spinner('Analyzing image...'):
        temp_path = "temp_img.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        fruit, status, confidence = get_prediction(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
    
    # Display results
    if fruit and status:
        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            if "Fresh" in status:
                st.markdown(f'<div class="fresh">🍏 {fruit} is {status}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rotten">🍎 {fruit} is {status}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
            
            # Add some fun facts
            if "Fresh" in status:
                st.success("This looks perfect for eating! 🍽️")
            else:
                st.warning("Consider discarding this item for better health 🚮")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
    <hr>
    <p>This AI classifier helps you determine the freshness of fruits and vegetables.</p>
    <p>Note: Results may vary based on image quality and lighting conditions.</p>
    </div>
    """, unsafe_allow_html=True)
