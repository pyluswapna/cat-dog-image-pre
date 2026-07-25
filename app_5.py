import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

# Load model
files = joblib.load("Dog_cat_image_prepro.pkl")
model = files["model"]

st.set_page_config(page_title="Dog vs Cat Image Classification")

st.title("🐶 Dog vs Cat Image Classification")
st.write("Upload an image to predict whether it is a Dog or a Cat.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to grayscale
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize
    img = cv2.resize(img, (28, 28))

    # Flatten
    img = img.flatten()

    # Convert to 2D array
    img = img.reshape(1, -1)

    # Prediction
    prediction = model.predict(img)[0]

    st.subheader("Prediction")

    if prediction == "dog":
        st.success("🐶 This is a Dog.")
    else:
        st.success("🐱 This is a Cat.")
    