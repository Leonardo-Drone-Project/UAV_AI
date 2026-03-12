# Multi Agent Swarm Coordination

## Overview

This module handles the swarm coordination side of the project.

The aim is to manage how multiple drones work together during the mission. It sits between the incoming drone state information and the Behaviour Tree mission logic.

At this stage, the module already provides a working baseline for:
- parent and child role election
- priority list generation
- parent loss detection
- parent reassignment
- converge completion checks
- BT input flag generation

## What this module does

The current coordination logic takes in the state of each drone in the swarm and uses that to decide:
- which drone should act as the parent
- which drones should act as children
- the child priority order
- whether the swarm is healthy enough to operate
- whether the parent has been lost
- whether a new parent has been assigned
- whether the swarm has converged on the target area

This gives the project a basic but working swarm coordination layer.

## Current decision logic

The current version uses a simple scoring approach for role election.

Each drone is scored using factors such as:
- battery state
- navigation health
- communications health
- target awareness

The highest scoring healthy drone becomes the parent.
The remaining healthy drones become children.
The child drones are then ordered into a priority list.

This keeps the first version simple and readable while still matching the main swarm logic needed by the Behaviour Tree.

## Current outputs

The current swarm decision output includes:
- selected parent drone
- child priority list
- assigned roles
- `swarm_coordinated`
- `roles_assigned`
- `priority_list_sent`
- `role_election_failed`
- `parent_lost`
- `parent_reassigned`
- `converge_complete`
- `target_known`
- estimated target position when available

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

This means the swarm module is already producing the main coordination signals needed by the Behaviour Tree.

## Current input structure

The module currently works from per-drone state inputs.

Each drone state includes values such as:
- drone ID
- position
- battery percentage
- communications health
- navigation health
- availability
- direct control state
- target detection state
- target position
- last update time

This is the base data used for role election, reassignment, and converge checks.

## Current files

`config.py`  
Stores swarm coordination settings such as role election weights and converge radius.

`models.py`  
Defines the drone state and swarm decision data structures.

`utils.py`  
Contains helper functions for scoring, target estimation, and distance checks.

`coordinator.py`  
Contains the main swarm coordination logic.

`interface.py`  
Provides a clean interface for the rest of the autonomy stack and the BT.

`main.py`  
Runs a local demo for role election, converge behaviour, and parent reassignment.

`README.md`  
Explains the purpose, structure, and current status of this module.

## Current status

The current baseline has already been tested locally.

The demo already shows that:
- a parent drone is selected correctly
- child drones are assigned correctly
- the priority list is generated correctly
- parent loss is detected correctly
- a new parent is assigned correctly
- BT coordination flags update correctly after reassignment

So the current module already gives the project a usable first swarm coordination baseline.

## What is still missing

This is still a first version. The core coordination logic is there, but more work is still needed before full system use.

The next main areas still to add are:
- heartbeat timeout handling using real update timestamps
- task assignment per drone
- target handover logic
- real comms message or payload flow
- stronger multi-drone deconfliction logic
- replay or run script for timestamped swarm testing
- integration with the full autonomy stack

## Why this matters

The Behaviour Tree already depends on swarm level decisions such as role assignment, reassignment, and converge completion.

Without a swarm coordination layer, the mission logic would not know:
- which drone is acting as the parent
- which drones should follow as children
- when to trigger reassignment
- when the drones have successfully converged on the target

This module fills that gap.

## Summary

This module is responsible for the coordination logic between multiple drones in the swarm.

In simplest terms:
- it decides parent and child roles
- it keeps track of priority order
- it detects when the parent has been lost
- it reassigns a new parent when needed
- it reports clean coordination flags to the Behaviour Tree

This makes it the core swarm coordination layer for the project.