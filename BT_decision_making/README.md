# Behaviour Tree (BT) Decision Making

## Overview
This module implements a Behaviour Tree–based decision layer for UAV autonomy.  
The Behaviour Tree determines *what the drone should do* at each timestep based on mission state, independent of perception and path planning.

At this stage, sensor inputs and actions are stubbed to allow deterministic testing of decision logic without relying on YOLO or A*.

---

## Behaviour States
The current Behaviour Tree supports the following high-level states:

- **SEARCH** – Slow yaw scan while holding position to locate a target.  
- **TRACK** – Actively track a detected target and keep it centred.  
- **LOST** – Short reacquisition state before returning to SEARCH.  
- **FAILSAFE** – Reserved for safety-critical conditions.

State transitions are logged for offline analysis.

---

## Architecture
The Behaviour Tree is implemented using a shared blackboard and modular components:


