import cv2
import numpy as np


class ErosionProcessor:
    def __init__(self):
        pass
    
    def apply_erosion(self, img, kernel_size=5, iterations=1):
        """
        Apply Erosion (Morphological Filtering)
        
        Args:
            img: Input image (should be grayscale or binary)
            kernel_size: Size of structuring element
            iterations: Number of times erosion is applied
            
        Returns:
            Eroded image
        """
        if img is None:
            return None
        
        # Convert to grayscale if image is color
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Create structuring element (kernel)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply erosion
        eroded = cv2.erode(gray, kernel, iterations=iterations)
        
        return eroded
