import cv2


class ThresholdingProcessor:
    def __init__(self):
        pass
    
    def apply_binary_threshold(self, img, threshold_value=127, max_value=255):
        """
        Apply Binary Thresholding
        
        Args:
            img: Input image (should be grayscale)
            threshold_value: Threshold value (0-255)
            max_value: Maximum value to use with THRESH_BINARY
            
        Returns:
            Thresholded image
        """
        if img is None:
            return None
        
        # Convert to grayscale if image is color
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
        
        return thresh
