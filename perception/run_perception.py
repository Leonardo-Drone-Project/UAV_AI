import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from perception.camera_interface import VideoFileCamera, OpenCVCamera, RealSenseCamera
from perception.yolo_interface import init_detector, detect


def make_run_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(args):
    # Camera selection
    if args.camera == "video":
        camera = VideoFileCamera(args.source)
    elif args.camera == "webcam":
        camera = OpenCVCamera(index=args.index, width=args.width, height=args.height, fps=args.fps)
    elif args.camera == "realsense":
        camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps, enable_depth=False)
    else:
        raise RuntimeError(f"Unknown camera mode: {args.camera}")

    # Outputs
    base_out = Path(args.out_dir).expanduser().resolve()
    run_dir = make_run_dir(base_out)
    latest_json = run_dir / "latest_detection.json"
    csv_path = run_dir / "detections.csv"

    # Detector init
    init_detector(
        weights_path=args.weights,
        device=args.device,
        conf_thres=args.conf,
        iou_thres=args.iou,
        prefer_classes=args.prefer_classes,
        min_box_px=args.min_box_px,
    )

    # CSV log
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(
        [
            "timestamp",
            "fps_ema",
            "detected",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "dx_px",
            "dy_px",
            "img_w",
            "img_h",
        ]
    )
    csv_f.flush()

    print("[INFO] Perception loop started")
    print(f"[INFO] Run folder: {run_dir}")

    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.1
    frame_i = 0

    try:
        while True:
            frame, timestamp = camera.get_frame()
            frame_i += 1

            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = inst_fps if fps_ema <= 0 else (1 - alpha) * fps_ema + alpha * inst_fps

            result = detect(frame, timestamp=timestamp)

            payload = {
                "timestamp": timestamp,
                "fps_ema": fps_ema,
                "result": result,
            }
            latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # CSV row
            if result["detected"]:
                x1, y1, x2, y2 = result["bbox"]
                dx, dy = result["offset_px"]
                cls = result["class"]
                conf = result["confidence"]
            else:
                x1 = y1 = x2 = y2 = 0
                dx = dy = 0
                cls = ""
                conf = 0.0

            img_w, img_h = result["image_size"]
            csv_w.writerow([timestamp, fps_ema, int(result["detected"]), cls, conf, x1, y1, x2, y2, dx, dy, img_w, img_h])

            if frame_i % args.flush_every == 0:
                csv_f.flush()

            if frame_i % args.print_every == 0:
                print(f"[INFO] fps={fps_ema:.1f} detected={int(result['detected'])} class={cls} conf={conf:.2f}")

            # Optional view, needs a display
            if args.view:
                vis = frame.copy()
                if result["detected"]:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{cls} {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                cv2.putText(
                    vis,
                    f"FPS {fps_ema:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Perception", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            time.sleep(1 / args.loop_hz)

    except KeyboardInterrupt:
        print("\n[INFO] Perception loop stopped")

    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        csv_f.flush()
        csv_f.close()
        print(f"[INFO] Saved: {csv_path}")
        print(f"[INFO] Saved: {latest_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--camera", choices=["video", "webcam", "realsense"], required=True)

    p.add_argument("--source", default="", help="Video path for --camera video")
    p.add_argument("--index", type=int, default=0, help="Webcam index for --camera webcam")

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--device", default="0", help='Ultralytics device string, "0" or "cpu"')
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--prefer-classes", nargs="*", default=["red_hat", "red_box"])
    p.add_argument("--min-box-px", type=int, default=20)

    p.add_argument("--loop-hz", type=float, default=30.0)
    p.add_argument("--out-dir", default="perception/outputs")
    p.add_argument("--print-every", type=int, default=60)
    p.add_argument("--flush-every", type=int, default=60)

    p.add_argument("--view", action="store_true")

    main(p.parse_args())