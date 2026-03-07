# EKF Sensor Fusion

## Overview

This part of the project focuses on sensor fusion using an Extended Kalman Filter, EKF.

The goal is to combine data from multiple onboard sensors into one reliable estimate of the drone state. Instead of trusting one sensor on its own, the EKF blends all available measurements to produce a better estimate of position, velocity, altitude, and heading.

This fused state estimate supports the rest of the autonomy stack, especially:
- Behaviour Tree mission logic
- A* navigation and guidance
- flight state monitoring
- logging and system validation

## What is sensor fusion

Sensor fusion is the process of combining measurements from different sensors to get a better estimate of the real system state.

Each sensor has strengths and weaknesses.

Examples:
- IMU gives fast motion data, but drifts over time
- GPS gives global position, but is noisy and slower
- barometer gives altitude information, but is affected by pressure changes
- magnetometer gives heading information, but is sensitive to magnetic disturbance

Sensor fusion uses all of these together so the final estimate is more accurate and more stable than any one sensor on its own.

## What is an Extended Kalman Filter

An Extended Kalman Filter is a state estimation algorithm used for nonlinear systems.

A standard Kalman Filter works for linear systems.
Drone motion is nonlinear, so the Extended Kalman Filter is used instead.

The EKF estimates the drone state over time by repeating two main steps:

1. Prediction  
   The filter predicts the next state using the system model.

2. Update  
   The filter corrects that prediction using new sensor measurements.

This cycle runs continuously while the drone is operating.

## Why EKF is used in this project

The drone system needs a reliable navigation state estimate for autonomous operation.

The EKF is useful here because it:
- combines multiple sensors into one fused estimate
- reduces noise
- handles sensor drift better than raw data
- provides smoother position and velocity estimates
- supports navigation and mission logic
- improves robustness when sensors are noisy

For this project, the EKF is intended to sit between the raw sensors and the rest of the system.

Flow:
raw sensors -> EKF fusion -> fused state estimate -> BT, A*, guidance, flight logic

## How the EKF works

### 1. State vector

The EKF tracks a set of variables called the state.

A typical drone navigation state may include:
- x position
- y position
- z position
- x velocity
- y velocity
- z velocity
- yaw or heading

Depending on the final design, extra states may also be added, such as:
- roll
- pitch
- sensor bias terms

### 2. Prediction step

The EKF first predicts the next state using a motion model.

This uses the previous estimated state and the system dynamics.

In simple terms:
- where was the drone before
- how was it moving
- where should it be now if that motion continues

This gives the predicted state.

### 3. Measurement step

The EKF then compares the predicted state with new sensor measurements.

Examples:
- GPS position
- IMU acceleration or angular motion
- barometer altitude
- magnetometer heading

The filter measures the difference between prediction and measurement.
This difference is called the innovation or residual.

### 4. Correction step

The EKF uses that difference to correct the predicted state.

The amount of correction depends on how much the filter trusts:
- the model
- each sensor measurement

This trust is controlled using uncertainty values.

### 5. Repeat

This process repeats continuously:
- predict
- measure
- correct
- repeat

Over time, the filter produces a fused estimate of the drone state.

## Basic EKF idea in simple terms

The EKF does not fully trust the model.
It also does not fully trust the sensors.

Instead, it balances both.

If a sensor is noisy, the EKF gives it less influence.
If the model starts drifting away from the measurements, the EKF pulls the estimate back toward the sensor data.

This is why EKF is widely used for navigation and robotics.

## Main outputs

The EKF is expected to provide fused navigation outputs for the rest of the system.

Typical outputs include:
- fused position
- fused velocity
- fused altitude
- fused heading
- navigation validity or health status
- optional uncertainty values or covariance terms

These outputs will support:
- Behaviour Tree state transitions
- A* path planning and guidance
- flight control integration
- mission progress checks

## Why this matters for the full system

The Behaviour Tree already depends on reliable navigation-related information.
Examples include:
- `nav_ok`
- `altitude_reached`
- `hover_stable`
- `arrived_search_area`
- `at_base`

The A* and guidance side also need reliable position information in a consistent frame.

Without a fused state estimate, the rest of the autonomy stack would rely on raw noisy sensor data, which is not ideal for autonomous mission execution.

## High-level structure of this module

This branch is intended to cover:
- EKF state prediction
- EKF measurement updates
- sensor fusion logic
- fused state output interface
- testing and validation of fused estimates

It should provide a clean interface for the rest of the project to read the estimated drone state.

## Summary

This module is responsible for turning raw navigation sensor data into a reliable fused drone state estimate.

In simple terms:
- sensor fusion combines multiple noisy sensor measurements
- the EKF predicts the drone state and then corrects it using incoming measurements
- the result is a more stable and more useful estimate of the drone state
- that estimate is then used by the rest of the autonomy system

This makes the EKF a core part of the autonomous navigation stack.