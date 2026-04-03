import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from numpy.linalg import norm

def test_feature_sim():
    model = load_model('knee.h5')
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    avg_feat = np.load('avg_knee_feature_vector.npy')
    avg_feat_norm = avg_feat / norm(avg_feat)
    
    def get_sim(img_arr):
        if img_arr.shape == (224, 224, 3):
            img_arr = np.expand_dims(img_arr, axis=0) / 255.0
        feat = feature_extractor.predict(img_arr)[0]
        return np.dot(feat, avg_feat_norm) / norm(feat)
        
    print("Testing similarities...")
    
    # 1. Dataset X-ray
    img = tf.keras.preprocessing.image.load_img('model/train/0Normal/NormalG0 (463).png', target_size=(224, 224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    print("Normal Test Image:", get_sim(img_arr))
    
    img = tf.keras.preprocessing.image.load_img('model/train/4Severe/SevereG4 (34).png', target_size=(224, 224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    print("Severe Test Image:", get_sim(img_arr))
    
    
    # 2. Random Noise
    noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    print("Noise:", get_sim(noise))
    
    # 3. Sketch-like (just simple shapes)
    sketch = np.ones((224, 224, 3), dtype=np.uint8) * 255
    # draw some black rectangles to simulate a sketch
    import cv2
    cv2.rectangle(sketch, (50, 50), (150, 150), (0, 0, 0), 5)
    cv2.circle(sketch, (100, 100), 30, (0, 0, 0), 2)
    print("Sketch:", get_sim(sketch))

if __name__ == '__main__':
    test_feature_sim()
