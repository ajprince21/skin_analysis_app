import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def predict_acne_type(image_path):
    # Load and Preprocess Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    model = load_model("models/acne_classification_model.h5")
    
    # Define Class Labels (Update as per your training dataset)
    class_labels = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Nodules", "Cysts"]
    # Predict Class
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)  # Get class index
    predicted_class_label = class_labels[predicted_class_index]  # Map to label
    confidence = round(np.max(prediction) * 100, 2)  # Confidence score
    return predicted_class_label, confidence