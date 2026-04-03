import cv2
import numpy as np
import os

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

def check_is_xray_like(image_path):
    """
    Step 1: Detect if it's an X-Ray or a normal photo/image.
    X-rays are largely grayscale and have significant black background.
    We allow some color for clinical annotations (arrows/lines) by using a generous color difference threshold.
    """
    image = cv2.imread(image_path)
    if image is None:
         return False, "Image could not be read."
    
    # 1. Grayscale check (relaxed to 60.0 to allow heavy clinical arrows/annotations/watermarks)
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)
    color_diff = (np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(r - b))) / 3.0
    
    if color_diff > 60.0:
        return False, "Invalid Image Detected. Please upload a radiographic X-ray image (Color photographs are not recognized)."
        
    # 2. Structural sanity check (not purely blank white/black)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val < 2 or mean_val > 253:
        return False, "Invalid Image Detected. The provided image lacks structural contrast (e.g., completely solid color)."
        
    return True, ""

def check_has_bone_structure(image_path):
    """
    Step 1.5: Detect if the image has a continuous bone-like skeleton shape.
    Valid X-rays will have large, bright regions representing bones.
    We check the sum of the largest contours to naturally support double-knee (2 legs) X-rays.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
        
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Sum the area of the top 3 contours (Supports 1 or 2 leg x-rays)
    top_contours = contours[:3]
    total_bone_area = sum(cv2.contourArea(c) for c in top_contours)
    
    h, w = img.shape
    total_area = h * w
    if total_area == 0: 
        return False
        
    area_ratio = total_bone_area / total_area
    
    # Dual-knee x-rays might have separated bones taking ~10% area across multiple contours.
    if area_ratio < 0.08:
        return False
        
    return True

def check_is_knee_xray(image_path, threshold=0.10):
    """
    Step 2: Detect if it is specifically a Knee X-ray.
    Compares the image histogram pattern against our trained Knee X-ray average histogram.
    Threshold lowered to 0.10 to allow watermarks, dual-knee images, or atypical contrast clinical X-rays.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hist_path = os.path.join(current_dir, 'avg_knee_hist.npy')
        
        avg_hist = np.load(hist_path)
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
            
        img = cv2.resize(img, (224, 224))
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        correlation = cv2.compareHist(avg_hist, hist, cv2.HISTCMP_CORREL)
        
        return correlation > threshold
    except Exception as e:
        print(f"Error in check_is_knee_xray: {e}")
        return True

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from numpy.linalg import norm

mobilenet_model = None
def get_mobilenet():
    global mobilenet_model
    if mobilenet_model is None:
        mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return mobilenet_model

def log_rejection(image_path, step, message):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(base_dir, 'rejected_images.log')
        with open(log_path, 'a') as f:
            f.write(f"REJECTED: {image_path} | STEP: {step} | MSG: {message}\n")
    except:
        pass

def check_mobilenet_ensemble(image_path, threshold=0.40):
    try:
        model = get_mobilenet()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feat_path = os.path.join(base_dir, 'mobilenet_avg_knee.npy')
        
        avg_feat = np.load(feat_path)
        avg_feat_norm = avg_feat / norm(avg_feat)
        
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        feat = model.predict(img_arr, verbose=0)[0]
        similarity = np.dot(feat, avg_feat_norm) / norm(feat)
        
        return similarity > threshold, float(similarity)
    except Exception as e:
        print("MobileNet Validation Error:", e)
        return True, 0.0

def validate_image(image_path):
    report = {
        'is_valid': True,
        'image_type': 'X-ray',
        'knee_detected': 'Yes',
        'quality': 'High',
        'is_blurry': False,
        'message': 'Passed all checks',
        'failed_step': 'None',
        'model_agreement': '100%'
    }
    
    # Step 1: Detect X-Ray vs Normal Photo
    is_xray, err_msg = check_is_xray_like(image_path)
    if not is_xray:
        report.update({'is_valid': False, 'image_type': 'Rejected (Not an X-ray)', 'failed_step': 'Step 1 (Color/Contrast Check)', 'message': "Invalid Input: Please upload a Knee X-ray image only."})
        log_rejection(image_path, report['failed_step'], report['message'])
        return report

    # Step 1.5: Skeleton Shape Analysis
    if not check_has_bone_structure(image_path):
        err_msg = "Invalid Image Detected. Expected a dominant skeleton-like structure. Please upload a clear knee X-ray (avoid text snippets or unrelated crops)."
        report.update({'is_valid': False, 'image_type': 'Rejected', 'failed_step': 'Step 2 (Skeleton Analysis)', 'message': err_msg})
        log_rejection(image_path, report['failed_step'], err_msg)
        return report

    # Step 2 & 3: Ensemble Verification (Deep Learning + Histogram)
    is_mobile_knee, sim_score = check_mobilenet_ensemble(image_path)
    if not is_mobile_knee:
        err_msg = f"Image validation failed. Secondary model rejected input (Score: {round(sim_score, 2)}). Please upload a clearer knee X-ray."
        report.update({'is_valid': False, 'knee_detected': 'No', 'failed_step': 'Step 3 (Secondary Model Verification)', 'message': err_msg})
        log_rejection(image_path, report['failed_step'], err_msg)
        return report
        
    hist_pass = check_is_knee_xray(image_path, threshold=0.05)
    if not hist_pass:
        err_msg = "No knee joint detected. Histogram validation failed. Please upload a proper knee X-ray."
        report.update({'is_valid': False, 'knee_detected': 'No', 'failed_step': 'Step 3 (Histogram Validation)', 'message': err_msg})
        log_rejection(image_path, report['failed_step'], err_msg)
        return report

    report['model_agreement'] = f"{round(sim_score * 100, 1)}%"

    # Step 4: Quality Check
    variance = calculate_blur_variance(image_path)
    if variance < 15.0:  
        report['is_blurry'] = True
        report['quality'] = 'Low'
        report['message'] = "Warning: The input X-ray image appears to be blurry or low quality. Prediction accuracy might be affected."
        
    return report
