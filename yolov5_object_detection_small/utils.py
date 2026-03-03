import torch
import cv2

def load_model():
    """
    Loads the pretrained YOLOv5s model from PyTorch Hub.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4  # Confidence threshold
    return model

def detect_and_draw(model, frame, target='person'):
    """
    Runs detection on a frame and draws bounding boxes for specified targets.
    """
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, det in detections.iterrows():
        class_name = det['name']
        if class_name == target or target == 'all':
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            conf = det['confidence']
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
