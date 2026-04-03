import os
import cv2
import numpy as np

def test_properties():
    test_dir = 'model/test'
    
    diffs = []
    variances = []
    edges = []
    
    count = 0
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                p = os.path.join(root, f)
                img = cv2.imread(p)
                if img is not None:
                    # Grayscale diff
                    b, g, r = cv2.split(img)
                    diff = (np.mean(np.abs(b-g)) + np.mean(np.abs(g-r)) + np.mean(np.abs(r-b))) / 3.0
                    diffs.append(diff)
                    
                    # Variance
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    variances.append(var)
                    
                    # Edges
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edge_img = cv2.Canny(blurred, 50, 150)
                    edge_density = np.sum(edge_img > 0) / (edge_img.shape[0] * edge_img.shape[1])
                    edges.append(edge_density)
                    
                    count += 1
                    if count >= 200:
                        break
        if count >= 200:
            break
            
    print("X-rays (200 images):")
    print(f"Diffs: Max {max(diffs):.2f}, Mean {np.mean(diffs):.2f}")
    print(f"Variance: Min {min(variances):.2f}, Mean {np.mean(variances):.2f}")
    print(f"Edge Density: Min {min(edges):.4f}, Max {max(edges):.4f}, Mean {np.mean(edges):.4f}")
    
    # Let's check a standard color image (we'll download or just use random color noise)
    noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    b, g, r = cv2.split(noise)
    print("\nColor Noise:")
    print("Diff:", (np.mean(np.abs(b-g)) + np.mean(np.abs(g-r)) + np.mean(np.abs(r-b))) / 3.0)
    
    # Grayscale noise
    gray_noise_1ch = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    gray_noise = cv2.merge([gray_noise_1ch, gray_noise_1ch, gray_noise_1ch])
    gray = cv2.cvtColor(gray_noise, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_img = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edge_img > 0) / (edge_img.shape[0] * edge_img.shape[1])
    print("\nGrayscale Noise Edges:", edge_density)
    
    # Blank images
    blank = np.zeros((224, 224), dtype=np.uint8)
    edge_img = cv2.Canny(blank, 50, 150)
    print("Blank Edges:", np.sum(edge_img))

if __name__ == '__main__':
    test_properties()
