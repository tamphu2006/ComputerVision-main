import cv2
import numpy as np

class GaussianProcessor:
    def __init__(self):
        pass

    def create_gaussian_kernel(self, ksize, sigma):
        """
        Create a 2D Gaussian kernel
        """
        k = ksize // 2
        x, y = np.mgrid[-k:k+1, -k:k+1]

        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel

    def apply_gaussian_filter(self, img, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian filtering using custom kernel
        """
        if img is None:
            raise ValueError("Input image is None")

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        kernel = self.create_gaussian_kernel(kernel_size, sigma)

        # Apply convolution
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img
