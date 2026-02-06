import cv2
import numpy as np


class LaplacianEdgeProcessor:
    def __init__(self):
        pass
    
    def apply_laplacian_edge(self, img, ksize=3):
        """
        Apply Laplacian Edge Detection
        
        Args:
            img: Input image (should be grayscale)
            ksize: Kernel size (1, 3, 5, or 7)
            
        Returns:
            Edge detected image
        """
        if img is None:
            return None
        
        # Convert to grayscale if image is color
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        
        # Convert back to uint8
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        return laplacian
