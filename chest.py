import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image


st.set_page_config(page_title="COVID‑19 X‑Ray Classifier", layout="centered")
st.title("🫁 COVID‑19 X‑Ray Classification")
st.write("Upload a chest X‑ray and click Predict.")

@st.cache_resource
def load_model():
    try:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='dense')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5, name='dropout')(x)
        predictions = Dense(3, activation='softmax', name='dense_1')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(r"C:\Users\hendy\Downloads\model_VGG16.h5", by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

st.success("✅ Model loaded (VGG16 preprocessing)")

uploaded_file = st.file_uploader("Choose an X‑ray image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X‑ray", width=300)

    # Preprocess using VGG16's preprocess_input (subtracts ImageNet mean)
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_processed = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_batch = np.expand_dims(img_processed, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            preds = model.predict(img_batch, verbose=0)[0]

        class_names = ['COVID‑19', 'Normal', 'Pneumonia']
        st.subheader("Prediction probabilities")

        cols = st.columns(3)
        for i, (cls, prob) in enumerate(zip(class_names, preds)):
            with cols[i]:
                if cls == 'COVID‑19':
                    st.error(f"🦠 {cls}")
                elif cls == 'Normal':
                    st.success(f"✅ {cls}")
                else:
                    st.warning(f"🔬 {cls}")
                st.write(f"**{prob*100:.2f}%**")
                st.progress(float(prob))

        idx = np.argmax(preds)
        diag = class_names[idx]
        conf = preds[idx] * 100
        st.markdown("---")
        if diag == 'COVID‑19':
            st.error(f"## 🦠 Diagnosis: {diag}")
        elif diag == 'Normal':
            st.success(f"## ✅ Diagnosis: {diag}")
        else:
            st.warning(f"## 🔬 Diagnosis: {diag}")
        st.write(f"Confidence: **{conf:.1f}%**")

        # Optional: show raw values for debugging
        with st.expander("Raw probabilities"):
            for cls, prob in zip(class_names, preds):
                st.write(f"{cls}: {prob:.6f}")