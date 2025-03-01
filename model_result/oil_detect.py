import cv2
import numpy as np

def calculate_oiliness(image_path):
    """
    Calculate the oiliness percentage of a single image using thresholding.

    Parameters:
    - image_path (str): File path of the image.

    Returns:
    - float: Oiliness percentage.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate oiliness percentage
    oily_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    oily_percentage = (oily_pixels / total_pixels) * 100

    return round(oily_percentage, 2)