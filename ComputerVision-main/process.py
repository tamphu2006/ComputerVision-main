import time
import cv2

from Week1_Capturering.Week1_captureSaveImg import CaptureSaveImgProcessor
from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor


class ImageProcessor:

    def __init__(self):
        self.capture = CaptureSaveImgProcessor()
        self.grayscale = GrayscaleProcessor()

    def process_frame(self, bgr_img):

        if bgr_img is None:
            raise ValueError("Input frame is None")

        # STEP 1: Capture & Save original image
        self.capture.capture_and_save_image(bgr_img, "test_capture.bmp")

        # STEP 2: Grayscale filter (MEASURE THIS)
        start_time = time.perf_counter()
        processed_img = self.grayscale.convert_to_grayscale(bgr_img)
        process_time_ms = (time.perf_counter() - start_time) * 1000

        # STEP 3: Save processed image
        self.capture.capture_and_save_image(
            processed_img,
            "processed_capture.bmp"
        )

        return processed_img, {}, process_time_ms
