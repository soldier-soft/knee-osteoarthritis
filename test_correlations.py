import os
import cv2
import numpy as np

def test_correlations():
    hist_path = 'utils/avg_knee_hist.npy'
    avg_hist = np.load(hist_path)
    
    test_dir = 'model/test'
    if not os.path.exists(test_dir):
        print("No test dir")
        return
        
    correlations = []
    
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                p = os.path.join(root, f)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    correlation = cv2.compareHist(avg_hist, hist, cv2.HISTCMP_CORREL)
                    correlations.append(correlation)
                    
    if correlations:
        print("Min correlation:", min(correlations))
        print("Max correlation:", max(correlations))
        print("Mean correlation:", np.mean(correlations))
        print("10th percentile:", np.percentile(correlations, 10))
        print("5th percentile:", np.percentile(correlations, 5))
        print("1st percentile:", np.percentile(correlations, 1))

if __name__ == '__main__':
    test_correlations()
