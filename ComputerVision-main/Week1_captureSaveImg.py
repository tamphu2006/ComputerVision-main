import cv2
import os
import numpy as np


class CaptureSaveImgProcessor:
    def __init__(self):
        pass
    
    def capture_and_save_image(self, bgr_img, filename):
        """
        Capture and save static image from camera
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            filename: Path to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if input image is valid
            if bgr_img is None or not isinstance(bgr_img, np.ndarray):
                return False

            # Create CapturedImage directory if it doesn't exist
            save_dir = "CapturedImage"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Combine path to save image
            save_path = os.path.join(save_dir, filename)

            # Save image
            success = cv2.imwrite(save_path, bgr_img)

            return success

        except Exception as e:
            print("Error saving image:", e)
            return False
