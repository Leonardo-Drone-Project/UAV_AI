from typing import Tuple
import numpy as np

class CameraInterface:
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """
        Returns:
            frame (np.ndarray): BGR image
            timestamp (float): time in seconds
        """
        raise NotImplementedError
