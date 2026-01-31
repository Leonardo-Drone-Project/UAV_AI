import argparse
import json
import time
from pathlib import Path

from perception.camera_interface import VideoFileCamera
from perception.yolo_interface import YoloDetector


OUTPUT_JSON = Path("perception/outputs/latest_detection.json")


def main(video_path: str):
    camera = VideoFileCamera(video_path)

    detector = YoloDetector(
        weights_path=r"C:\Users\Stewy\UAV_AI\yolov5_object_detection_small\runs\red_object_detect2\weights\best.pt",
        device="0",  # GPU
        conf_thres=0.25,
        iou_thres=0.45,
        prefer_classes=["red_hat", "red_box"],
    )

    print("[INFO] Perception loop started")

    try:
        while True:
            frame, timestamp = camera.get_frame()

            result = detector.detect(frame)

            payload = {
                "timestamp": timestamp,
                "result": result,
            }

            OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

            # Simulate real-time (≈30 Hz)
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n[INFO] Perception loop stopped")

    finally:
        camera.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to video file (camera stub)",
    )
    args = parser.parse_args()

    main(args.source)
