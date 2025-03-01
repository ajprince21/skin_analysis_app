import cv2
import numpy as np

def load_image(image_path):
    """Loads an image using OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return cv2.resize(image, (256, 256))

def get_redness_score(image):
    """Calculates redness intensity in the skin region."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    redness_score = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    return round(redness_score / 255 * 100, 2)  # Convert to %

def get_pigmentation_score(image):
    """Detects pigmentation by analyzing darker regions in grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:50])
    pigmentation_score = dark_pixels / (image.shape[0] * image.shape[1])
    return round(pigmentation_score * 100, 2)  # Convert to %

def get_pores_score(image):
    """Estimates pores by detecting fine textures using High-Pass Filtering."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_pass = cv2.Laplacian(gray, cv2.CV_64F)
    high_pass = np.abs(high_pass)
    pores_score = np.mean(high_pass) / 255
    return round(pores_score * 100, 2)  # Convert to %

def details_scores(image_path):
    # Load and process image
    image = load_image(image_path)
    # Get scores
    scores = []
    redness = get_redness_score(image)
    pigmentation = get_pigmentation_score(image)
    pores = get_pores_score(image)
    scores.append(redness)
    scores.append(pigmentation)
    scores.append(pores)
    return scores