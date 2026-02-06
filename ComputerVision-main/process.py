import time
import cv2
import numpy as np
import os

from Week1_Capturing.Week1_captureSaveImg import CaptureSaveImgProcessor
from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor
from Week2_Filtering.Week2_Ex2_Gaussian import GaussianProcessor
from Week2_Filtering.Week2_Ex3_MedianBlur import MedianBlurProcessor
from Week2_Filtering.Week2_Ex4_SobelEdge import SobelEdgeProcessor
from Week2_Filtering.Week2_Ex5_LaplacianEdge import LaplacianEdgeProcessor
from Week2_Filtering.Week2_Ex6_Sharpening import SharpeningProcessor
from Week2_Filtering.Week2_Ex7_BilateralFilter import BilateralFilterProcessor
from Week2_Filtering.Week2_Ex8_Thresholding import ThresholdingProcessor
from Week2_Filtering.Week2_Ex9_Erosion import ErosionProcessor
from Week2_Filtering.Week2_Ex10_Dilation import DilationProcessor

class ImageProcessor:
    """
    Class for processing images from camera feed
    Implements computer vision techniques from ProjectProgress.txt
    """
    
    def __init__(self):
        """Initialize image processor with calibration parameters"""
        self.camera_matrix = None  # Camera calibration matrix
        self.dist_coeffs = None    # Distortion coefficients
        self.homography_matrix = None  # Homography transformation matrix
        self.previous_frame = None  # For motion detection
        self.tracked_objects = []   # For object tracking
        pass
    
    
    # =============================================================================
    # STEP 10: SYSTEM INTEGRATION (Week 14)
    # Topic: All Course Concepts
    # =============================================================================
    
    def process_frame(self, bgr_img, filter_name='grayscale'):


        if bgr_img is None:
            raise ValueError("Input frame is None")
        
        start_time = time.perf_counter()
        results = {}
        
        ###################### FILTER PROCESSING PIPELINE #########################
        
        # Initialize processors
        saveImg = CaptureSaveImgProcessor()
        
        # Apply selected filter
        if filter_name == 'grayscale':
            processor = GrayscaleProcessor()
            processed_img = processor.convert_to_grayscale(bgr_img)
            
        elif filter_name == 'gaussian':
            processor = GaussianProcessor()
            # Convert to grayscale first
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            processed_img = processor.apply_gaussian_filter(gray)
            
        elif filter_name == 'median':
            processor = MedianBlurProcessor()
            # Convert to grayscale first
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            processed_img = processor.apply_median_blur(gray)
            
        elif filter_name == 'sobel':
            processor = SobelEdgeProcessor()
            # Sobel automatically converts to grayscale
            processed_img = processor.apply_sobel_edge(bgr_img, direction='x')
            
        elif filter_name == 'laplacian':
            processor = LaplacianEdgeProcessor()
            # Laplacian automatically converts to grayscale
            processed_img = processor.apply_laplacian_edge(bgr_img)
            
        elif filter_name == 'sharpening':
            processor = SharpeningProcessor()
            # Convert to grayscale first
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            processed_img = processor.apply_sharpening(gray)
            
        elif filter_name == 'bilateral':
            processor = BilateralFilterProcessor()
            # Convert to grayscale first
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            processed_img = processor.apply_bilateral_filter(gray)
            
        elif filter_name == 'threshold':
            processor = ThresholdingProcessor()
            # Threshold automatically converts to grayscale
            processed_img = processor.apply_binary_threshold(bgr_img)
            
        elif filter_name == 'erosion':
            processor = ErosionProcessor()
            # Erosion automatically converts to grayscale
            processed_img = processor.apply_erosion(bgr_img)
            
        elif filter_name == 'dilation':
            processor = DilationProcessor()
            # Dilation automatically converts to grayscale
            processed_img = processor.apply_dilation(bgr_img)
            
        else:
            # Default to grayscale
            processor = GrayscaleProcessor()
            processed_img = processor.convert_to_grayscale(bgr_img)
        
        #################################################################################
        
        process_time_ms = (time.perf_counter() - start_time) * 1000
        return processed_img, results, process_time_ms
    
    def visualize_results(self, bgr_img, results):
        """
        Visualize all processing results on image
        
        Args:
            bgr_img: Original image
            results: Dictionary of results from process_frame
            
        Returns:
            Annotated image
        """

        pass
