- The YOLOv5s model is a lightweight, real-time object detection model from the YOLOv5 family, developed by Ultralytics. 

- It balances speed and performance, making it ideal for edge devices and UAV-based inference where compute resources and latency are constrained.

- It should detect and localise humans, trees, and other objects in an open-field environment via drone-mounted cameras.

- This will serve as the first stage in a detect-then-track pipeline, feeding into a tracker for persistent UAV target tracking.

## Reproducibility: Training and Testing Commands

This section documents the exact commands and configuration used to train and evaluate the YOLO-based red target detection model so that the results can be fully reproduced.

Activate the Python virtual environment before running any commands:

```bash
venv_gpu\Scripts\activate

Training is started by running:
python main.py --mode train

After training, the model was evaluated on recorded video footage at multiple resolutions to analyse detection performance versus processing speed. Testing is run using:
python main.py --mode test --video ../videos/red_object_test1.mov

During video playback in testing mode, the ESC key can be pressed to exit.
