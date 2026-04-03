import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

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

def make_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
                
        if not last_conv_layer_name: return None
                
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Safe normalization to prevent dividing by 0
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0.0:
            return None
            
        heatmap = tf.maximum(heatmap, 0) / max_val
        return heatmap.numpy()
    except Exception as e:
        print("GradCAM Error:", e)
        return None

def generate_heatmap_image(img_path, heatmap):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + img * 0.6
        
        filename = os.path.basename(img_path)
        heatmap_path = img_path.replace(filename, 'heatmap_' + filename)
        cv2.imwrite(heatmap_path, superimposed_img)
        
        return heatmap_path
    except:
        return img_path

def predict_label(img_path):
    """
    Predicts the label for the given image path and returns the stage, confidence, and heatmap path.
    """
    if model is None:
        load_ml_model()

    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    img_array = test_image.reshape(1, 224, 224, 3)

    predict_x = model.predict(img_array)
    classes_x = np.argmax(predict_x, axis=1)
    
    predicted_class_index = classes_x[0]
    predicted_label = verbose_name[predicted_class_index]
    
    confidence = float(predict_x[0][predicted_class_index]) * 100
    
    heatmap = make_gradcam_heatmap(img_array, model)
    if heatmap is not None:
        heatmap_path_full = generate_heatmap_image(img_path, heatmap)
    else:
        heatmap_path_full = img_path
        
    return predicted_label, round(confidence, 2), heatmap_path_full
