import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("yolov5/runs/train/red_object_detect/weights/best.pt")

# Set video input
video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

# Test these resolutions
resolutions = [(1280, 720), (854, 480), (640, 360), (426, 240)]  # 720p, 480p, etc.

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
    print(f"\nTesting resolution: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized = cv2.resize(frame, (width, height))

        # Run detection
        results = model(resized)

        # Plot + show
        annotated = results[0].plot()
        cv2.imshow(f"YOLOv5 - {width}x{height}", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

cap.release()





