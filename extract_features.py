import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

def extract_features():
    model = load_model('knee.h5')
    # Get the model architecture to find the penultimate layer
    print([layer.name for layer in model.layers])
    
    # We will create a feature extractor model
    # Usually the layer before the final dense layer is a Flatten or GlobalAveragePooling2D or Dense layer
    layer_name = model.layers[-2].name
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    train_dir = 'model/train/0Normal'
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)[:50]]
    
    features = []
    for f in files:
        img = tf.keras.preprocessing.image.load_img(f, target_size=(224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        feat = feature_extractor.predict(img_arr)
        features.append(feat[0])
        
    avg_feature = np.mean(features, axis=0)
    np.save('avg_knee_feature_vector.npy', avg_feature)
    print("Saved average feature vector of shape:", avg_feature.shape)

if __name__ == '__main__':
    extract_features()
