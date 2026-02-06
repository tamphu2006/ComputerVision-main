import cv2
import numpy as np


class SobelEdgeProcessor:
    def __init__(self):
        pass
    
    def apply_sobel_edge(self, img, direction='x'):
        """
        Apply Sobel Edge Detection
        
        Args:
            img: Input image (should be grayscale)
            direction: 'x' for horizontal edges, 'y' for vertical edges, 'both' for combined
            
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
        
        if direction == 'x':
            # Sobel X (vertical edges)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        elif direction == 'y':
            # Sobel Y (horizontal edges)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        else:  # both
            # Combine both directions
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Convert back to uint8
        sobel = np.absolute(sobel)
        sobel = np.uint8(sobel)
        
        return sobel
