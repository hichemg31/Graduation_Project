import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    # Convert img1 to grayscale if needed
    if img1.ndim == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1

    # img2 already grayscale
    g2 = img2

    # Resize if needed
    if g1.shape != g2.shape:
        g1 = cv2.resize(g1, (g2.shape[1], g2.shape[0]))

    # Ensure float32
    g1 = g1.astype(np.float32)
    g2 = g2.astype(np.float32)

    # Normalize to [0,1] if needed
    if g1.max() > 1.0:
        g1 /= 255.0
    if g2.max() > 1.0:
        g2 /= 255.0

    return ssim(g1, g2, data_range=1.0)
