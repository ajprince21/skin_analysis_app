import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load Pretrained Model (You Need to Train This First)


def predict_dry_skin(image_path):
    # Load and Preprocess Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict Dryness
    model = load_model("models/dry_skin_model.h5")
    prediction = model.predict(img)[0][0]  # Get probability
    dry_percentage = round(prediction * 100, 2)
    return dry_percentage