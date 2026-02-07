import cv2
import numpy as np
from config import IMAGE_SIZE, FEATURE_SIZE

def preprocess_for_ssim(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64))  # or even 32×32
    return small

def preprocess_frame(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img
