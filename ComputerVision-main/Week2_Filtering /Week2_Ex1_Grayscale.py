import cv2
import numpy as np


class GrayscaleProcessor:
    """
    Class for processing images from camera feed
    """
    frame = None  # BGR numpy array
    def __init__(self, frame=None):
        """Initialize image processor"""
        self.frame = frame
        pass
    def convert_to_grayscale(self, bgr_image):
        if bgr_image is None:
            return None
        
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return gray_img
    
