# Multi Agent Swarm Coordination

## Overview

This module handles the swarm coordination side of the project.

The current focus is:
- role election
- parent and child assignment
- priority list generation
- parent loss detection
- parent reassignment
- converge completion logic
- clean BT input flags

This module sits above the vehicle state inputs and below the Behaviour Tree mission logic.

## Current outputs

The current coordination logic outputs:
- selected parent drone
- child priority list
- assigned roles
- swarm_coordinated
- roles_assigned
- priority_list_sent
- role_election_failed
- parent_lost
- parent_reassigned
- converge_complete

## BT integration

The interface exposes these BT related flags:
- swarm_coordinated
- roles_assigned
- priority_list_sent
- role_election_failed
- parent_lost
- parent_reassigned
- converge_complete

## Current files

`config.py`  
Swarm coordination settings.

`models.py`  
Drone state and swarm decision data classes.

`utils.py`  
Helper functions for scoring, target estimation, and distance checks.

`coordinator.py`  
Core swarm coordination logic.

`interface.py`  
Clean interface for the rest of the autonomy stack and BT.

`main.py`  
Simple local demo script.

## Notes

This first version keeps the coordination logic simple and readable.

It gives the project a working baseline for:
- role election
- reassignment
- BT flag generation

Later work should add:
- real comms message flow
- real task distribution
- real target handover
- stronger deconfliction logic
- integration with the full autonomy stack