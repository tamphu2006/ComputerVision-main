import cv2


class BilateralFilterProcessor:
    def __init__(self):
        pass
    
    def apply_bilateral_filter(self, img, d=9, sigma_color=75, sigma_space=75):
        """
        Apply Bilateral Filter (edge-preserving smoothing)
        
        Args:
            img: Input image (grayscale or color)
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        if img is None:
            return None
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        return filtered
