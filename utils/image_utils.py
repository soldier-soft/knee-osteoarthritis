import cv2
import numpy as np

def calculate_blur_variance(image_path):
    """
    Computes the Laplacian variance of the image which acts as a blur metric.
    Lower variance = blurrier image.
    """
    image = cv2.imread(image_path)
    if image is None:
         return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def check_is_grayscale(image_path, threshold=20):
    """
    Heuristic check if the image is mostly grayscale (like typical X-rays).
    It compares the color channels to check if they are roughly equal.
    """
    image = cv2.imread(image_path)
    if image is None:
         return False
    
    b, g, r = cv2.split(image)
    
    # Calculate absolute differences between color channels
    diff_bg = np.mean(np.abs(b - g))
    diff_gr = np.mean(np.abs(g - r))
    diff_rb = np.mean(np.abs(r - b))
    
    # If the differences are very low, the image is mostly grayscale
    average_diff = (diff_bg + diff_gr + diff_rb) / 3.0
    
    return average_diff < threshold

def validate_image(image_path):
    """
    Validates if an uploaded image is valid for prediction.
    Returns: (is_valid, is_blurry, message)
    """
    is_valid = True
    is_blurry = False
    message = ""
    
    # 1. Check if it's grayscale (is it likely an X-ray?)
    if not check_is_grayscale(image_path):
        is_valid = False
        message = "Invalid Image. Please upload an X-ray image (color images are not supported)."
        return (is_valid, is_blurry, message)
        
    # 2. Check for blur
    variance = calculate_blur_variance(image_path)
    if variance < 100:  # Adjust threshold based on testing
        is_blurry = True
        message = "Warning: The image uploaded seems to be low quality or blurry. The prediction results might be inaccurate."
        
    return (is_valid, is_blurry, message)
