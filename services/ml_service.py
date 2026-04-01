import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = None
verbose_name = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe"
}

def load_ml_model(model_path='knee.h5'):
    global model
    if model is None:
        model = load_model(model_path)
    return model

def predict_label(img_path):
    """
    Predicts the label for the given image path and returns the stage and confidence score.
    """
    # Load model if not loaded
    if model is None:
        load_ml_model()

    # Preprocessing
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    # Prediction
    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    
    predicted_class_index = classes_x[0]
    predicted_label = verbose_name[predicted_class_index]
    
    # Calculate confidence %
    confidence = float(predict_x[0][predicted_class_index]) * 100
    
    return predicted_label, round(confidence, 2)
