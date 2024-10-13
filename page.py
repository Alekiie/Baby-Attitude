import tensorflow as tf
# import load_model
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


st.set_page_config(page_title="Baby Attitude Classification", page_icon="🧊", layout="centered", initial_sidebar_state="expanded")
st.title("Baby Attitude Classification")


st.markdown('---')
st.subheader("Upload an image of a baby to classify the attitude of the baby")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.markdown('---')

model=load_model(os.path.join("models", "imageclassifier.h5"))

# Function to preprocess and predict the class of the image
def predict_image(img):
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Resize the image to the required input size of the model
    resize = tf.image.resize(img_array, (256, 256))
    
    # Normalize the pixel values
    resize = resize / 255.0
    
    # Expand dimensions to fit the model input shape
    yhat = model.predict(np.expand_dims(resize, axis=0))
    
    
    # Return the class prediction
    if yhat > 0.5: 
        return 'Sad Baby'
    else:
        return 'Happy Baby'

# If an image is uploaded
if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    
    # Make a prediction
    pred = predict_image(img)
    
    # Display the prediction
    st.subheader(f"Predicted: {pred}!")
    
    # Show the uploaded image
    st.image(img, use_column_width=True)

# Add a footer watermark centered at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 50%;
        bottom: 0;
        transform: translateX(-50%);
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-color: #f1f1f1;
        color: #555;
    }
    </style>
    <div class="footer">
        <p>Built with ☕, too many bug fixes, and a dash of magic by Alexander 🧙‍♂️</p>
        <p>Curious? Explore my digital playground at <a href="https://github.com/alekiie" target="_blank">GitHub</a> 🕵️‍♂️</p>
    </div>
    """,
    unsafe_allow_html=True
)
