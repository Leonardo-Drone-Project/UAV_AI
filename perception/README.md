# Perception Module for YOLO

This folder is the project's **single interface** for object detection.

## What it outputs (per frame / per image)
- `target_detected` (bool)
- `target_conf` (float)
- `bbox_xyxy` and `bbox_cxcywh`
- `target_offset_px = (dx, dy)` from image centre

These outputs are what the **Behaviour Tree** will use later.


