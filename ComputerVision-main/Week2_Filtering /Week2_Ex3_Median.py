import cv2

class MedianProcessor:
    def __init__(self):
        pass

    def apply_median_filter(self, img, kernel_size=5):
        if img is None:
            raise ValueError("Input image is None")

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        return cv2.medianBlur(img, kernel_size)
