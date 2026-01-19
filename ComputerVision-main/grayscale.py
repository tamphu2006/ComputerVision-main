import cv2
import numpy as np

class GrayscaleProcessor:
    """Class for grayscale image processing"""
    
    def __init__(self):
        pass
    
    def convert_to_grayscale(self, bgr_img):
        """Convert BGR image to grayscale"""
        if bgr_img is None:
            raise ValueError("Input image is None")
        
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def apply_threshold(self, gray_img, threshold=127):
        """Apply binary threshold to grayscale image"""
        _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def apply_adaptive_threshold(self, gray_img):
        """Apply adaptive threshold"""
        adaptive = cv2.adaptiveThreshold(
            gray_img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return adaptive
