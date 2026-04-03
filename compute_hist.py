import cv2
import os
import numpy as np

hists = []
train_dir = 'e:/projects/ZE/JPDL12-A Novel Method to Predict Knee Osteoarthritis/JPDL12-A Novel Method to Predict Knee Osteoarthritis/SOURCE CODE/Knee Osteoarthritis/model/train'
dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir)]
count = 0

for d in dirs:
    files = os.listdir(d)
    for f in files[:50]:
        p = os.path.join(d, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hists.append(hist)
            count += 1

avg_hist = np.mean(hists, axis=0)
np.save('avg_knee_hist.npy', avg_hist)
print(f'Saved avg hist from {count} images')
