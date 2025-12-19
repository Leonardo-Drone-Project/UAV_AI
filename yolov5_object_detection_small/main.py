import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


def train_red_object():
    """
    Train YOLO on the red_box / red_hat dataset exported from Roboflow.
    Run with: python main.py --mode train
    """
    # Load base YOLO model (your yolov5su.pt in UAV_AI root)
    model = YOLO("../yolov5su.pt")

    # Train on your dataset
    model.train(
        data="../datasets/red_object/data.yaml",  # path to your YAML
        imgsz=640,
        epochs=100,
        batch=8,          # reduced from 16 to lower memory use
        workers=0,        # IMPORTANT: no multiprocessing -> avoids spawn error
        project="runs",   # output folder
        name="red_object_detect2",
        # device=0,       # uncomment to force GPU 0 
    )

def test_resolutions(video_path: str):
    """
    Test trained model on the same video at multiple resolutions.
    Run with: python main.py --mode test --video path/to/video.mp4
    """
    # Load trained model (path relative to this main.py file)
    model = YOLO("runs/red_object_detect2/weights/best.pt")

    resolutions = [
        (1280, 720),
        (854, 480),
        (640, 360),
        (426, 240),
    ]

    conf = 0.25

    for (width, height) in resolutions:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = f"annotated_{width}x{height}.mp4"
        out = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

        print(f"\nTesting: {width}x{height} -> saving {out_path}")

        frame_count = 0
        det_counter = Counter()
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            results = model.predict(resized, conf=conf, verbose=False)
            r = results[0]

            # Count detections by class id
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                det_counter.update(cls_ids)

            annotated = r.plot()
            out.write(annotated)

            cv2.imshow(f"YOLO - {width}x{height}", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        t1 = time.time()
        elapsed = max(t1 - t0, 1e-6)
        fps_proc = frame_count / elapsed

        print(f"Processed frames: {frame_count}")
        print(f"Processing FPS:   {fps_proc:.2f}")
        print("Detections per class id:", dict(det_counter))
        print("Class names:", model.names)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="test",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="path/to/your/video.mp4",
        help="Path to input video for testing",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_red_object()
    else:
        test_resolutions(args.video)






