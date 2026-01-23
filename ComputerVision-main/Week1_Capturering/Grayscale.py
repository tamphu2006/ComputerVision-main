import time
import cv2
import numpy as np


class ImageProcessor:
    """
    Class for processing images from camera feed
    """
    frame = None  # BGR numpy array
    def __init__(self, frame=None):
        """Initialize image processor"""
        self.frame = frame
        pass
    
    def process_frame(self, bgr_img):
        """
        Process a single frame
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            
        Returns:
            tuple: (Processed image, process time in ms)
        """
        if bgr_img is None:
            raise ValueError("Input frame is None")

        processed_img = self.convert_to_grayscale(bgr_img)

        start_time = time.perf_counter()
        process_time_ms = (time.perf_counter() - start_time) * 1000

        return processed_img, process_time_ms
    def preprocess(self, bgr_img):
        """
        Preprocess image (e.g., resize, normalize)
        
        Args:
            bgr_img: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        # TODO: Implement preprocessing
        pass
    
    def postprocess(self, result):
        """
        Postprocess results
        
        Args:
            result: Processed result
            
        Returns:
            Postprocessed result
        """
        # TODO: Implement postprocessing
        pass
    
    def convert_to_grayscale(self, bgr_image):
        if bgr_image is None:
            return None
        
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return gray_img
    
