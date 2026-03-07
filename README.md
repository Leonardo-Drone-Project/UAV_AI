# UAV_AI Project
- This is where all the AI algorithms will be developed
- I will commit and push onto here every so often so you all can check as I am working through it if curious
- please don't make any changes without telling me first as it can mess things up

---
# System Architecture

The UAV_AI system is designed using a modular layered architecture to separate perception, decision-making, planning, and state estimation.  
Benefits of this are improves scalability, maintainability, hardware portability, and testing.

---

## 1. Perception Layer

The **Perception Layer** is responsible for detecting mission-relevant targets from camera input.

### Input
- RGB/BGR image frame from onboard camera

### Output (Structured Detection Object)
- `target_detected` (bool)
- `confidence` (float)
- `bounding_box` (x1, y1, x2, y2)
- `pixel_offset` (dx, dy from image centre)

This structured output is hardware-agnostic and feeds directly into the Behaviour Tree and motion control logic.

### Design Goals
- GPU-accelerated inference (CUDA)
- Clean interface: `detect(frame)`
- No camera logic inside detection module
- Ready for embedded deployment (Jetson)

---

## 2. Decision Layer – Behaviour Tree

The **Behaviour Tree (BT)** manages high-level autonomous behaviour through mode switching.

### Example Modes
- Search
- Target Tracking
- Return to Home
- Emergency Landing

The BT continuously evaluates:
- Perception results
- System state
- Mission conditions

It then determines which behaviour should be active.

### Why Behaviour Trees?
- Clear hierarchical logic
- Modular state transitions
- Safe fallback handling
- Easy extension with new behaviours

The BT acts as the bridge between perception and motion planning.

---

## 3. Planning Layer – A* Path Planning

The system uses the **A\* algorithm** for path optimisation and waypoint routing.

### Cost Function

f(n) = g(n) + h(n)

Where:
- `g(n)` = path cost so far
- `h(n)` = heuristic estimate to goal

### Applications
- Obstacle-aware navigation
- Waypoint optimisation
- Target-biased routing
- Return-to-home logic

A\* enables computationally efficient and optimal route generation.

---

## 4. Sensor Fusion – Extended Kalman Filter (EKF)

An **Extended Kalman Filter (EKF)** fuses multiple sensor streams:

- IMU
- GPS
- Velocity estimates
- other optionals

### Benefits
- Improved position accuracy
- Better orientation estimation
- Noise rejection
- Robust state estimation in dynamic environments

Reliable state estimation is essential for stable autonomous flight.

---

## 5. Multi-Agent Coordination (MAS)

The **Multi-Agent System module** supports swarm-level coordination.

### Capabilities
- Cooperative search
- Task allocation
- Collision avoidance
- Shared situational awareness

This allows the system to scale from a single UAV to coordinated multi-drone missions.

---

## 6. Predictive Maintenance – Random Forest

A **Random Forest classifier** is used for anomaly detection and maintenance prediction.

### Inputs
- Sensor logs
- Vibration data
- Battery performance
- Operational metrics

### Outputs
- Fault classification
- Maintenance alerts

This improves mission reliability and operational safety.

---

# Object Detection Module

The system uses a fine-tuned **YOLO model** trained on custom classes:

- `red_box`
- `red_hat`

---

## Detection Pipeline

1. Camera frame acquisition
2. YOLO inference (GPU accelerated)
3. Bounding box extraction
4. Target selection (highest confidence or preferred class)
5. Offset calculation from image centre
6. Structured output generation

---

## Detection Output Structure

Each inference produces:

- `class_id`
- `class_name`
- `confidence`
- `bbox_xyxy`
- `bbox_cxcywh`
- `pixel_offset`

The **pixel offset** is critical for:

- Gimbal control
- Target centering
- Behaviour Tree switching (Search → Track)
- A\* waypoint biasing

---

## Design Principles

- Hardware abstraction via camera interface
- Strict separation between perception and decision logic
- GPU-compatible for Jetson deployment
- Fully testable without hardware
- Behaviour Tree–ready output format
