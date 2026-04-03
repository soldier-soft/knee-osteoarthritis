import cv2
import numpy as np

def is_knee_xray(image_path, threshold=0.5):
    avg_hist = np.load('avg_knee_hist.npy')
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
        
    img = cv2.resize(img, (224, 224))
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    correlation = cv2.compareHist(avg_hist, hist, cv2.HISTCMP_CORREL)
    return correlation

p1 = 'e:/projects/ZE/JPDL12-A Novel Method to Predict Knee Osteoarthritis/JPDL12-A Novel Method to Predict Knee Osteoarthritis/SOURCE CODE/Knee Osteoarthritis/model/train/0Normal/NormalG0 (100).png'
cv2.imwrite('black.png', np.zeros((224, 224), dtype=np.uint8))
cv2.imwrite('white.png', np.ones((224, 224), dtype=np.uint8)*255)
cv2.imwrite('noise.png', np.random.randint(0, 256, (224, 224), dtype=np.uint8))

print('Knee x-ray:', is_knee_xray(p1))
print('Black:', is_knee_xray('black.png'))
print('White:', is_knee_xray('white.png'))
print('Noise:', is_knee_xray('noise.png'))

