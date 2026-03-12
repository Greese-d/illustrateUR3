import cv2
import numpy as np


class PortraitProcessor:

    def __init__(self, threshold=130):
        """
        threshold: value used for binary thresholding
        """
        self.threshold = threshold

    def process(self, image):
        """
        Convert input image into binary image (0 or 1).

        Parameters
        ----------
        image : numpy.ndarray
            Input image (BGR or grayscale)

        Returns
        -------
        binary_image : numpy.ndarray
            Image containing only 0 and 1
        """

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Slight blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to binary (0 or 255)
        _, binary = cv2.threshold(
            blurred,
            self.threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Convert 255 -> 1
        #binary = (binary > 0).astype(np.uint8)
        binary = (binary > 0).astype(np.uint8) * 255
        binary = 255 - binary  # Invert back to have foreground as white (255) and background as black (0)

        return binary