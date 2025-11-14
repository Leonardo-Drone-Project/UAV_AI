import torch
import cv2
from ultralytics import YOLO

# ✅ Load YOLOv5s model
model = YOLO("yolov5s.pt")  # Automatically downloads if not found

# ✅ Check for CUDA (GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# ✅ Open video stream (0 = default webcam, replace with drone stream if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # ✅ Run detection
    results = model(frame)

    # ✅ Annotate detections
    annotated_frame = results[0].plot()

    # ✅ Show results
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # ⏹ Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



