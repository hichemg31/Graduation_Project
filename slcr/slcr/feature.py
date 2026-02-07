import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, size=32):
        self.size = size
        self.dim = size * size

    def extract(self, img):
        # img is already preprocessed (float [0,1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.size, self.size))
        feat = small.flatten().astype(np.float32)
        return feat
