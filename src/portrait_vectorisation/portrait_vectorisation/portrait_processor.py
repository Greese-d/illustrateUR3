import cv2
import numpy as np
import mediapipe as mp


class PortraitProcessor:

    def __init__(self):
        # Initialize MediaPipe segmentation
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

    def remove_background(self, image):
        """Remove background using MediaPipe segmentation."""

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.segmenter.process(rgb)

        mask = results.segmentation_mask

        # Convert to binary
        mask = (mask > 0.4).astype(np.uint8)

        # Smooth mask
        kernel = np.ones((7,7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Blur mask edges
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15,15), 0)
        mask = mask > 0.3

        output = image.copy()

        # Set background to white
        output[~mask] = 255

        return output

    def process(self, image):
        """
        Full portrait processing pipeline
        """

        # Remove background
        img = self.remove_background(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Smooth noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        #filtered = cv2.GaussianBlur(gray, (5,5), 0)

        # # Detect edges
        edges = cv2.Canny(filtered, 20, 50)
        #edges = cv2.medianBlur(edges, 3)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        smooth_contours = []

        for c in contours:
        
            if cv2.arcLength(c, False) < 20:
                continue
            
            epsilon = 3.0
            smooth = cv2.approxPolyDP(c, epsilon, False)

            smooth_contours.append(smooth)

        canvas = np.zeros_like(edges)
        cv2.drawContours(canvas, smooth_contours, -1, 255, 1)

        return 255 - canvas