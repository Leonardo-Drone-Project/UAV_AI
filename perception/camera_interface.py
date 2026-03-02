from __future__ import annotations

from typing import Tuple, Optional
import time

import cv2
import numpy as np


class CameraInterface:
    """
    Abstract camera interface.
    get_frame returns:
      frame (np.ndarray): BGR image
      timestamp (float): time in seconds
    """

    def get_frame(self) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def release(self) -> None:
        return


class VideoFileCamera(CameraInterface):
    """
    Camera stub that reads frames from a video file.
    Loops video on end.
    """

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_frame(self) -> Tuple[np.ndarray, float]:
        ok, frame = self.cap.read()

        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from video")

        ts = time.time()
        return frame, ts

    def release(self) -> None:
        if self.cap:
            self.cap.release()


class OpenCVCamera(CameraInterface):
    """
    Webcam via OpenCV index.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index: {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

    def get_frame(self) -> Tuple[np.ndarray, float]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read from OpenCV camera")
        ts = time.time()
        return frame, ts

    def release(self) -> None:
        if self.cap:
            self.cap.release()


class RealSenseCamera(CameraInterface):
    """
    Intel RealSense color stream via pyrealsense2.
    Returns BGR frames.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, enable_depth: bool = False):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError("pyrealsense2 import failed. Install pyrealsense2.") from e

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))

        self.enable_depth = bool(enable_depth)
        self.align = None
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
            self.align = rs.align(rs.stream.color)

        self.pipeline.start(self.config)

    def get_frame(self) -> Tuple[np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)

        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense returned no color frame")

        frame = np.asanyarray(color.get_data())
        ts = time.time()
        return frame, ts

    def release(self) -> None:
        if self.pipeline:
            self.pipeline.stop()


