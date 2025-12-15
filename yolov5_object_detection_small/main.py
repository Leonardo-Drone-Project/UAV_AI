import cv2
from ultralytics import YOLO

# Load your trained YOLOv5 model
model = YOLO("yolov5/runs/train/red_object_detect/weights/best.pt")  # Adjust path if needed

# Path to test video (or use 0 for webcam)
video_path = "your_video.mp4"  # <- replace with actual path or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Define test resolutions (you can adjust these)
resolutions = [
    (1280, 720),  # 720p
    (854, 480),   # 480p
    (640, 360),   # 360p
    (426, 240)    # 240p
]

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
    print(f"\n--- Testing at resolution: {width}x{height} ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (width, height))

        # Perform inference
        results = model(resized)

        # Draw annotations
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow(f"Detection at {width}x{height}", annotated_frame)

        # Press 'q' to move to next resolution
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

cap.release()




