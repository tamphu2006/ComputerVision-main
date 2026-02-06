import cv2


class MedianBlurProcessor:
    def __init__(self):
        pass
    
    def apply_median_blur(self, img, kernel_size=5):
        """
        Apply Median Blur filtering to reduce noise
        
        Args:
            img: Input image (grayscale or color)
            kernel_size: Size of kernel (must be odd, e.g., 3, 5, 7)
            
        Returns:
            Filtered image
        """
        if img is None:
            return None
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        filtered_img = cv2.medianBlur(img, kernel_size)
        return filtered_img
