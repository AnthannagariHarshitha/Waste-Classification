import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ Waste Classification using CNN")
st.write("Upload a waste image to classify it as **Biodegradable** or **Non-Biodegradable**.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()
class_names = ["Biodegradable", "Non-Biodegradable"]

def predict_image(img):
    image = img.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 64, 64, 3))
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    label, confidence = predict_image(image)
    st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")
