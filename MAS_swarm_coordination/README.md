# Multi Agent Swarm Coordination

## Overview

This module handles the swarm coordination side of the project.

The aim is to manage how multiple drones work together during the mission. It sits between the incoming drone state information and the Behaviour Tree mission logic.

At the current stage, the module already supports:
- parent and child role election
- priority list generation
- timeout-based parent loss detection
- heartbeat-based drone validity checks
- parent reassignment
- target-aware task assignment
- target owner selection
- target handover detection
- target handover completion
- multi-drone deconfliction flags
- converge completion checks
- BT input flag generation
- replay-based swarm testing
- payload adapter support for incoming drone state updates
- structured outbound command message generation

## What this module does

The coordination logic takes in the state of each drone in the swarm and uses that to decide:
- which drone should act as the parent
- which drones should act as children
- the child priority order
- whether the swarm is healthy enough to operate
- whether the parent has been lost
- whether a new parent has been assigned
- what task each drone should currently perform
- which drone should own the target
- whether target tracking should be handed over
- whether drones are too close and need deconfliction
- whether the swarm has converged on the target area

This gives the project a working swarm coordination layer that already matches the main Behaviour Tree coordination flow.

## Message flow

The module now has a stricter input and output structure.

Incoming payloads are converted into validated `DroneStatusMessage` objects, then converted into internal `DroneState` objects.

Outgoing swarm decisions can be converted into structured `SwarmCommandMessage` objects for downstream modules.

This gives the swarm module a cleaner contract for real system integration.

## Current decision logic

The current version uses a scoring approach for role election.

Each drone is scored using:
- battery state
- navigation health
- communications health
- target awareness

The highest scoring healthy drone becomes the parent.
The remaining healthy drones become children.
The child drones are then ordered into a priority list.

Drone freshness is checked using `last_update_s`.
Heartbeat health is checked using:
- `last_heartbeat_s`
- `missed_heartbeats`

This means role election and reassignment depend on both state health and message freshness.

## Parent and child roles

The parent role is used for:
- swarm coordination
- target reporting
- high-level task distribution

The child role is used for:
- target converge
- target tracking
- search support

The parent and the target tracking drone are not forced to be the same drone.

## Task assignment

When a target is known, the current task logic assigns:
- parent to coordination and reporting
- one selected drone to target tracking
- remaining child drones to target converge

When no target is known, the current task logic assigns:
- parent to swarm coordination
- one child to primary search
- remaining children to support search

If deconfliction is active, one or more drones can be assigned:
- `hold_position_deconflict`

## Target ownership and handover

The current version tracks:
- current target owner
- target tracking drone
- whether handover is required
- whether handover has completed

The handover logic uses:
- target distance
- drone health
- drone freshness
- heartbeat validity
- battery threshold
- target confidence
- tracking lock timeout
- role preference for child tracking

This keeps target ownership stable while still allowing a better drone to take over when needed.

## Deconfliction

The current version also adds a first deconfliction layer.

It checks pairwise drone separation and flags when drones are too close.
This produces:
- `deconfliction_active`
- `collision_risk`
- `deconfliction_pairs`

The current task logic then blocks lower-priority drones with `hold_position_deconflict` if needed.

This is a simple first version of deconfliction, not a full collision avoidance system.

## Current outputs

The current swarm decision output includes:
- selected parent drone
- child priority list
- assigned roles
- task assignments
- stale drone IDs
- heartbeat-lost drone IDs
- `swarm_coordinated`
- `roles_assigned`
- `priority_list_sent`
- `role_election_failed`
- `parent_lost`
- `parent_reassigned`
- `converge_complete`
- `target_known`
- estimated target position when available
- `target_owner_id`
- `target_tracking_drone_id`
- `target_handover_required`
- `target_handover_complete`
- `deconfliction_active`
- `collision_risk`
- `deconfliction_pairs`

## BT integration

The interface already exposes clean BT related flags.

Current BT related outputs are:
- `swarm_coordinated`
- `roles_assigned`
- `priority_list_sent`
- `role_election_failed`
- `parent_lost`
- `parent_reassigned`
- `converge_complete`
- `target_handover_required`
- `target_handover_complete`
- `deconfliction_active`
- `collision_risk`

## Current files

`config.py`  
Stores swarm coordination settings such as role election weights, timeout values, handover rules, and separation limits.

`models.py`  
Defines the internal drone state and swarm decision data structures.

`schemas.py`  
Defines the validated input and output message schemas.

`utils.py`  
Contains helper functions for scoring, target estimation, freshness checks, heartbeat checks, target ownership, deconfliction, task assignment, and distance checks.

`coordinator.py`  
Contains the main swarm coordination logic.

`interface.py`  
Provides a clean interface for the rest of the autonomy stack and the BT.

`adapters.py`  
Converts incoming payloads into validated message objects and internal drone state objects, and converts decisions into outbound command messages.

`main.py`  
Runs a local demo for role election, target handover, deconfliction, and parent reassignment.

`run_swarm.py`  
Runs timestamped swarm replay input through the coordinator and logs the outputs.

`README.md`  
Explains the purpose, structure, and current status of this module.

## Current status

The current module has already been tested locally.

The demo shows that:
- a parent drone is selected correctly
- child drones are assigned correctly
- the priority list is generated correctly
- target-aware task assignment works
- target owner selection works
- target handover works
- stale drones are detected correctly
- heartbeat-based invalid drones can be identified
- deconfliction flags activate correctly
- parent timeout is detected correctly
- a new parent is assigned correctly after timeout
- BT coordination flags update correctly after reassignment, handover, and deconfliction

## What is still missing before full drone use

This is now a strong coordination core, but some work is still needed before full system use.

The main remaining areas are:
- real inter-drone message transport
- acknowledgement and retry logic for coordination messages
- stronger conflict resolution than simple hold-position logic
- integration with live vehicle states and live tracking updates
- scenario testing with real project-shaped replay data
- final integration with the full autonomy stack

## Summary

This module is responsible for the coordination logic between multiple drones in the swarm.

In simple terms:
- it decides parent and child roles
- it keeps track of priority order
- it assigns high-level tasks
- it tracks stale and heartbeat-lost drones
- it detects when the parent has been lost
- it reassigns a new parent when needed
- it selects the target owner
- it handles target handover
- it flags deconfliction risk
- it produces command-style outputs for downstream modules
- it reports clean coordination flags to the Behaviour Tree

This makes it the core swarm coordination layer for the project.