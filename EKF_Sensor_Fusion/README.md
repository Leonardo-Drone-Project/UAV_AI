# EKF Sensor Fusion

## Overview

This part of the project focuses on sensor fusion using an Extended Kalman Filter, EKF.

The purpose of the EKF is to combine measurements from multiple onboard sensors into one fused navigation state estimate. Instead of relying on a single sensor, the filter combines all available information to produce a more stable estimate of the drone position, velocity, altitude, and attitude.

For the current version of this project, the EKF is built around:
- GPS
- IMU
- RealSense

This fused state estimate supports the rest of the autonomy stack, especially:
- Behaviour Tree mission logic
- A* navigation and waypoint guidance
- flight state monitoring
- system logging and validation

## What is sensor fusion

Sensor fusion is the process of combining measurements from different sensors to estimate the real state of the system more accurately than any single sensor on its own.

Each sensor has useful information, but each one also has limits.

Examples:
- IMU gives fast motion information, but drifts over time
- GPS gives absolute position, but is noisy and slower
- RealSense provides local position or motion correction, but only works well under certain conditions

Sensor fusion combines these strengths and reduces the effect of their individual weaknesses.

## What is an Extended Kalman Filter

An Extended Kalman Filter is a state estimation algorithm used for nonlinear systems.

A standard Kalman Filter is designed for linear systems. Drone motion is nonlinear, so the Extended Kalman Filter is used instead.

The EKF works in two repeated stages:

1. Prediction  
   The filter predicts the next state using the previous estimate and the motion model.

2. Update  
   The filter corrects that prediction using new sensor measurements.

This process runs continuously while the drone is active.

## Why EKF is used in this project

The drone needs a reliable navigation state estimate for autonomous operation.

The EKF is useful here because it:
- combines multiple sensors into one fused estimate
- reduces the effect of measurement noise
- limits drift better than raw sensor data alone
- provides smoother position and velocity estimates
- supports navigation, mission logic, and guidance
- improves robustness when one sensor becomes noisy or unreliable

In this project, the EKF sits between the raw sensors and the rest of the autonomy system.

Flow:
raw sensors -> EKF fusion -> fused state estimate -> BT, A*, guidance, flight logic

## Current sensor setup

The current EKF version is designed around three sensor sources:

### GPS
Used as the main absolute position source.

### IMU
Used for high-rate prediction of motion using acceleration and angular rate.

### RealSense
Used as a local correction source for position, velocity, or yaw when available.

The intended fusion idea is:
- IMU drives the fast prediction step
- GPS corrects long-term position drift
- RealSense corrects local motion or pose drift

## How the EKF works

### 1. State vector

The EKF tracks a set of internal state variables.

The current implementation uses:
- x position
- y position
- z position
- x velocity
- y velocity
- z velocity
- roll
- pitch
- yaw

This gives a 9-state navigation estimate.

### 2. Prediction step

The EKF predicts the next state using the IMU input and the previous estimate.

In simple terms, it asks:
- where was the drone before
- how fast was it moving
- how is it rotating
- where should it be after the next short time step

This produces the predicted state.

### 3. Measurement update step

The EKF then compares the predicted state with new sensor measurements.

In the current version:
- GPS updates position
- RealSense updates position, velocity, and yaw when available

The filter calculates the difference between the predicted state and the measured state. This difference is called the residual.

### 4. Correction step

The EKF uses the residual to correct the estimate.

The amount of correction depends on how much the filter trusts:
- the motion model
- each sensor measurement
- the current uncertainty

This is controlled by the covariance matrices and noise settings.

### 5. Repeat

This process repeats continuously:
- predict
- update
- correct
- repeat

Over time, the filter produces a fused navigation estimate.

## Basic EKF idea in simple terms

The EKF does not fully trust the model.
It also does not fully trust the sensors.

Instead, it balances both.

If a measurement is noisy, the filter gives it less influence.
If the prediction starts drifting away from the measurements, the filter pulls the estimate back.

This is why EKF is widely used in robotics, UAV navigation, and autonomous systems.

## Main outputs

The current EKF interface provides a fused navigation state with:
- fused x, y, z position
- fused vx, vy, vz velocity
- fused roll, pitch, yaw attitude
- timestamp
- position variance
- velocity variance
- attitude variance
- navigation health flag, `nav_ok`

These outputs are intended to support:
- Behaviour Tree mission checks
- A* navigation and waypoint following
- flight-side integration
- logging and validation

## BT related outputs

The EKF itself estimates fused navigation state.

A thin interface layer then derives BT useful navigation flags from that fused state.

Current derived flags include:
- `nav_ok`
- `altitude_reached`
- `hover_stable`
- `arrived_search_area`
- `at_base`

This keeps the EKF focused on estimation while still making the output useful for mission logic.

## Current module structure

The current EKF folder contains:

### `config.py`
Defines filter settings, noise values, covariance settings, and health thresholds.

### `state.py`
Defines the state layout and fused output format.

### `measurements.py`
Defines the measurement structures for:
- IMU
- GPS
- RealSense

### `ekf.py`
Contains the EKF prediction and update logic.

### `interface.py`
Provides the clean fused state interface for the rest of the autonomy stack, including BT-related navigation flags.

### `demo_ekf.py`
Provides a local test using simulated GPS, IMU, and RealSense measurements.

### `README.md`
Explains the purpose, design, and usage of this module.

## Current status

The current EKF version is working as a first usable baseline.

At the moment it already supports:
- IMU driven prediction
- GPS position updates
- RealSense position, velocity, and yaw correction
- fused state output
- navigation health checking
- derived BT navigation flags
- local demo testing

The demo has already been run successfully and shows:
- stable fused position estimates
- stable fused velocity estimates
- valid navigation health output
- correct BT-style derived navigation flags

## Why this matters for the full system

The Behaviour Tree already depends on reliable navigation-related information.
Examples include:
- `nav_ok`
- `altitude_reached`
- `hover_stable`
- `arrived_search_area`
- `at_base`

The A* and guidance side also need reliable position and motion data in a consistent frame.

Without a fused state estimate, the rest of the autonomy stack would rely on noisy raw sensor data, which is not suitable for autonomous mission execution.

## High-level scope of this branch

This branch covers:
- EKF state prediction
- EKF measurement updates
- sensor fusion logic
- fused state output interface
- derived BT navigation flags
- testing and validation of fused estimates

This branch is intended to cover both the EKF fusion itself and the interface that makes the fused navigation state usable by the rest of the autonomy stack.

## Summary

This module is responsible for turning raw GPS, IMU, and RealSense data into a reliable fused drone navigation state estimate.

In simple terms:
- sensor fusion combines multiple noisy sensor measurements
- the EKF predicts the drone state and then corrects it using incoming measurements
- the result is a more stable and more useful navigation estimate
- that estimate is then used by the rest of the autonomy system

This makes the EKF a core part of the autonomous navigation stack.