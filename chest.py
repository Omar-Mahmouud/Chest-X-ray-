import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_VGG16.keras")

classes = ["COVID", "Normal", "Viral Pneumonia"]

st.title("COVID-19 X-ray Classification App")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    st.write("### Prediction:", predicted_class)