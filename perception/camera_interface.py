import cv2
import time
from typing import Tuple
import numpy as np


class VideoFileCamera(CameraInterface):
    """
    Camera stub that reads frames from a video file.
    Acts exactly like a real camera for the perception loop.
    """

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_frame(self) -> Tuple[np.ndarray, float]:
        ret, frame = self.cap.read()

        # Loop video when it ends (useful for testing)
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from video")

        timestamp = time.time()
        return frame, timestamp

    def release(self) -> None:
        self.cap.release()

