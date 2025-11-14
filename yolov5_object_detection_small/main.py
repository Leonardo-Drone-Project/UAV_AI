import cv2
import torch
from utils import load_model, detect_and_draw

# Initialize webcam (replace 0 with your drone feed if needed)
cap = cv2.VideoCapture(0)

# Load YOLOv5 small model
model = load_model()

# Target class
TARGET_CLASS = 'person'

print("[INFO] Starting detection loop...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera not returning frame. Exiting.")
        break

    # Run detection and draw results
    result_frame = detect_and_draw(model, frame, target=TARGET_CLASS)

    # Display the results
    cv2.imshow('UAV Object Detection Feed', result_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
