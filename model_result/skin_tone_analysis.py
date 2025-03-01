import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

def load_image(image_path):
    """Loads an image using OpenCV, with fallback to PIL."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is not None:
        return image

    try:
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception:
        return None

def extract_skin(image):
    """Extracts skin region from the image using YCrCb color space."""
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(img_ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    return cv2.bitwise_and(image, image, mask=mask)

def get_dominant_color(image, k=3):
    """Finds the dominant color in the extracted skin region using KMeans clustering."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    img = img[np.any(img != [0, 0, 0], axis=1)]  # Remove black pixels

    if len(img) == 0:
        return None

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)
    return kmeans.cluster_centers_[np.argmax(np.unique(kmeans.labels_, return_counts=True)[1])].astype(int)

def classify_skin_tone(rgb_color):
    """Classifies the detected skin tone into predefined categories."""
    skin_tones = {
        "Fair": (255, 224, 189),
        "Medium": (204, 136, 94),
        "Dark": (102, 51, 0),
    }

    return min(skin_tones, key=lambda tone: np.linalg.norm(np.array(rgb_color) - np.array(skin_tones[tone])))

def skin_tone_analysis(image_path):
    # Run the skin tone detection
    image = load_image(image_path)

    if image is not None:
        skin_region = extract_skin(image)
        dominant_skin_color = get_dominant_color(skin_region)

        if dominant_skin_color is not None:
            skin_tone = classify_skin_tone(dominant_skin_color)
            print(f"Detected Skin Tone: {skin_tone}")
            return skin_tone
        else:
            print("No skin detected in the image.")
            return False
    else:
        print("Failed to process the image.")
        return False