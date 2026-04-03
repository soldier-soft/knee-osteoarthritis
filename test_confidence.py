import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_model_confidence():
    model = load_model('knee.h5')
    
    test_dir = 'model/test'
    confidences = []
    
    files_to_check = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                files_to_check.append(os.path.join(root, f))
    
    # Check 100 random test images
    np.random.shuffle(files_to_check)
    for p in files_to_check[:100]:
        img = tf.keras.preprocessing.image.load_img(p, target_size=(224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        pred = model.predict(img_arr)
        conf = np.max(pred[0])
        confidences.append(conf)
        
    print("Test images confidences:")
    print("Min:", min(confidences))
    print("Max:", max(confidences))
    print("Mean:", np.mean(confidences))
    print("10th percentile:", np.percentile(confidences, 10))
    print("5th percentile:", np.percentile(confidences, 5))
    
    # Create fake noise/random images
    print("\nFake images confidences:")
    fake_confs = []
    for _ in range(20):
        noise = np.random.rand(1, 224, 224, 3)
        pred = model.predict(noise)
        fake_confs.append(np.max(pred[0]))
    print("Noise Mean:", np.mean(fake_confs))
    print("Noise Max:", np.max(fake_confs))

    # Test black/white
    black = np.zeros((1, 224, 224, 3))
    print("Black Conf:", np.max(model.predict(black)[0]))
    
    white = np.ones((1, 224, 224, 3))
    print("White Conf:", np.max(model.predict(white)[0]))

if __name__ == '__main__':
    test_model_confidence()
