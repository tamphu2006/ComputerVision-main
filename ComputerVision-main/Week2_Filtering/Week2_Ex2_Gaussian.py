import cv2


class GaussianProcessor:
    def __init__(self):
        pass
    
    def apply_gaussian_filter(self, img, kernel_size=(5, 5), sigma=1.0):
        """
        Apply Gaussian filtering to reduce noise
        
        Args:
            img: Input image (grayscale or color)
            kernel_size: Size of Gaussian kernel (must be odd, e.g., 3, 5, 7)
            sigma: Standard deviation
            
        Returns:
            Filtered image
        """
        if img is None:
            return None

        # Apply Gaussian Blur
        filtered_img = cv2.GaussianBlur(img, kernel_size, sigma)
        return filtered_img
