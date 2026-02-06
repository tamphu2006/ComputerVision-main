import cv2
import numpy as np


class SharpeningProcessor:
    def __init__(self):
        pass
    
    def apply_sharpening(self, img):
        """
        Apply Sharpening Filter to enhance edges
        
        Args:
            img: Input image (grayscale or color)
            
        Returns:
            Sharpened image
        """
        if img is None:
            return None
        
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply filter
        sharpened = cv2.filter2D(img, -1, kernel)
        
        return sharpened
