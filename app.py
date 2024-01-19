import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
import requests
from io import BytesIO
import os
from cvzone import stackImages
import streamlit as st
from PIL import Image

model_a = load_model('A_model_7.h5')
model_b = load_model('B_model_7.h5')

def process_image(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_layer = img_lab[:, :, 0]
    l_layer = np.expand_dims(l_layer, axis=0)

    a_layer = model_a.predict(l_layer)
    b_layer = model_b.predict(l_layer)

    a_layer = np.squeeze(a_layer, axis=0)
    b_layer = np.squeeze(b_layer, axis=0)

    l_layer = l_layer.reshape((128, 128))
    a_layer = a_layer.reshape((128, 128))
    b_layer = b_layer.reshape((128, 128))

    output_image = tf.stack([l_layer, b_layer, a_layer], axis=-1)
    output_image = np.array(output_image)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_LAB2BGR)
    output_image = cv2.resize(output_image, (200, 200))

    return output_image

def main():
    st.title("Image Processing Streamlit App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.resize(image, (128, 128))

        # Display original image
        st.image(image, caption="Original Image", use_column_width=False)

        # Process the image
        processed_image = process_image(image)

        # Display processed image
        st.image(processed_image, caption="Processed Image", use_column_width=False)

if __name__ == "__main__":
    main()
