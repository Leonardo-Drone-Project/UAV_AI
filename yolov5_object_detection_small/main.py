import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt  # NEW: for plotting


def train_red_object():
    """
    Train YOLO on the red_box / red_hat dataset exported from Roboflow.
    Run with: python main.py --mode train
    """
    model = YOLO("../yolov5su.pt")

    model.train(
        data="../datasets/red_object/data.yaml",
        imgsz=640,
        epochs=100,
        batch=8,          # reduced from 16 to lower memory use
        workers=0,        # avoids multiprocessing issue on Windows
        project="runs",
        name="red_object_detect2",
    )


def test_resolutions(video_path: str):
    """
    Test trained model on the same video at multiple resolutions and
    create summary plots.
    Run with: python main.py --mode test [--video path/to/video]
    """
    model = YOLO("runs/red_object_detect2/weights/best.pt")

    resolutions = [
        (1920, 1080),   # Full HD
        (1280, 720),
        (854, 480),
    ]

    conf = 0.25

    # For plotting later
    res_labels = []
    fps_list = []
    avg_dets_list = []

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
            key = cv2.waitKey(1)
            if key == 27:  # ESC to quit
                break

        t1 = time.time()
        elapsed = max(t1 - t0, 1e-6)
        fps_proc = frame_count / elapsed

        total_dets = sum(det_counter.values())
        avg_dets_per_frame = total_dets / frame_count if frame_count > 0 else 0.0

        # Print summary for this resolution
        print(f"Processed frames: {frame_count}")
        print(f"Processing FPS:   {fps_proc:.2f}")
        print(f"Total detections: {total_dets}")
        print(f"Avg detections/frame: {avg_dets_per_frame:.3f}")
        print("Class names:", model.names)

        # Store for plots
        res_labels.append(f"{width}x{height}")
        fps_list.append(fps_proc)
        avg_dets_list.append(avg_dets_per_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # ---- PLOTS ----
    # 1) FPS vs resolution
    plt.figure()
    plt.plot(res_labels, fps_list, marker="o")
    plt.xlabel("Resolution")
    plt.ylabel("Processing FPS")
    plt.title("YOLO Processing FPS vs Resolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fps_vs_resolution.png")

    # 2) Avg detections per frame vs resolution
    plt.figure()
    plt.plot(res_labels, avg_dets_list, marker="o")
    plt.xlabel("Resolution")
    plt.ylabel("Average detections per frame")
    plt.title("YOLO Detections vs Resolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("detections_vs_resolution.png")

    print("\nSaved plots:")
    print(" - fps_vs_resolution.png")
    print(" - detections_vs_resolution.png")


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
        default="C:/Users/Stewy/UAV_AI/videos/red_object_test1.mov",
        help="Path to input video for testing",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_red_object()
    else:
        test_resolutions(args.video)








