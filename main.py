import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import base64
import requests
from PIL import Image
from io import BytesIO

# Config
st.set_page_config(page_title="🖼 Image Caption Generator", layout="centered")

# Style
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .block-container {
        max-width: 800px;
        margin: auto;
        padding-top: 2rem;
    }
    h1 {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00c6ff;
        color: black;
        transform: scale(1.05);
    }
    .caption-box {
        background-color: #222;
        color: #ffff66;
        font-weight: bold;
        padding: 1rem;
        margin-top: 1.5rem;
        border-radius: 12px;
        border: 2px solid #ffff66;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🖼 Image Caption Generator")

# Paths
MODEL_PATH = "models/model.keras"
FEATURE_EXTRACTOR_PATH = "models/feature_extractor.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
IMG_SIZE = 224
MAX_LENGTH = 34

@st.cache_resource
def load_resources():
    caption_model = load_model(MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

caption_model, feature_extractor, tokenizer = load_resources()

def generate_caption(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH)
        yhat = caption_model.predict([features, seq], verbose=0)
        yhat_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_idx, None)
        if word is None or word == "endseq":
            break
        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Upload section
option = st.radio("📤 Upload Method", ["Upload from Device", "Upload from URL"], horizontal=True)
temp_filename = "uploaded_temp.jpg"
image_ready = False

if option == "Upload from Device":
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_ready = True

elif option == "Upload from URL":
    url = st.text_input("Paste image URL here:")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(temp_filename)
            image_ready = True
        except:
            st.error("Couldn't load image from the URL.")

# Centered image display
if image_ready:
    img_bytes = open(temp_filename, "rb").read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <img src="data:image/jpeg;base64,{encoded}" width="300" style="border-radius:10px;" />
        </div>
    """, unsafe_allow_html=True)

    # Centered generate button
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 20px;">', unsafe_allow_html=True)
    generate = st.button("✨ Generate Caption")
    st.markdown('</div>', unsafe_allow_html=True)

    if generate:
        with st.spinner("Generating..."):
            caption = generate_caption(temp_filename)
            st.markdown(f"<div class='caption-box'>📝 Caption:<br>{caption}</div>", unsafe_allow_html=True)
            os.remove(temp_filename)

# Info panel
with st.expander("📚 How It Works"):
    st.markdown("""
    - *CNN* extracts image features.
    - *LSTM* generates text from visual features.
    - Trained on paired datasets like MS-COCO.
    """)
