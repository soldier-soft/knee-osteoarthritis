import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from numpy.linalg import norm

def test_mobilenet_features():
    print("Loading MobileNetV2...")
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    train_dir = 'model/train/0Normal'
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)[:50]]
    
    print("Extracting features from 50 training images...")
    features = []
    for f in files:
        img = tf.keras.preprocessing.image.load_img(f, target_size=(224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        feat = model.predict(img_arr)[0]
        features.append(feat)
        
    avg_feat = np.mean(features, axis=0)
    avg_feat_norm = avg_feat / norm(avg_feat)
    
    np.save('mobilenet_avg_knee.npy', avg_feat)
    
    def get_sim(img_arr):
        if img_arr.shape == (224, 224, 3):
            img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        feat = model.predict(img_arr)[0]
        return np.dot(feat, avg_feat_norm) / norm(feat)
        
    print("\nTesting similarities...")
    
    img = tf.keras.preprocessing.image.load_img('model/train/0Normal/NormalG0 (463).png', target_size=(224, 224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    print("Dataset Knee:", get_sim(img_arr))
    
    noise = np.random.randint(0, 256, (224, 224, 3)).astype(np.float32)
    print("Noise:", get_sim(noise))
    
    sketch = np.ones((224, 224, 3), dtype=np.float32) * 255
    import cv2
    cv2.rectangle(sketch, (50, 50), (150, 150), (0, 0, 0), 5)
    print("Sketch:", get_sim(sketch))
    
    black = np.zeros((224, 224, 3), dtype=np.float32)
    print("Black:", get_sim(black))

if __name__ == '__main__':
    test_mobilenet_features()
