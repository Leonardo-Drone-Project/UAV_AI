# Multi Agent Swarm Coordination

## Overview

This module handles the swarm coordination side of the project.

The aim is to manage how multiple drones work together during the mission. It sits between the incoming drone state information and the Behaviour Tree mission logic.

At the current stage, the module supports:
- parent and child role election
- priority list generation
- timeout-based parent loss detection
- heartbeat-based drone validity checks
- parent reassignment
- target-aware task assignment
- target owner selection
- target handover detection
- target handover completion
- target owner stability locking
- parent reassignment stability locking
- multi-drone deconfliction flags
- converge completion checks
- BT input flag generation
- replay-based swarm testing
- payload adapter support for incoming drone state updates
- structured outbound command message generation
- unit test coverage for the main coordination behaviours

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

This gives the project a working swarm coordination layer that matches the main Behaviour Tree coordination flow.

## Message flow

The module uses a stricter input and output structure.

Incoming payloads are converted into validated `DroneStatusMessage` objects, then converted into internal `DroneState` objects.

Outgoing swarm decisions are converted into structured `SwarmCommandMessage` objects for downstream modules.

This gives the swarm module a clean contract for real system integration.

## Current decision logic

The module uses a scoring approach for role election.

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

## Frozen task vocabulary

The task names are now fixed and should not be changed:

- `coordinate_swarm`
- `coordinate_and_report_target`
- `coordinate_report_and_track_target`
- `search_primary`
- `search_support`
- `track_target`
- `converge_target`
- `hold_position_deconflict`

These should be treated as the final command vocabulary for the MAS module.

## Target ownership and handover

The module tracks:
- current target owner
- target tracking drone
- pending target owner
- handover state
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

A stability lock is also applied after successful handover so the system does not immediately reassign target ownership again.

A stability lock is also applied after parent reassignment so the system does not immediately churn target ownership during the transition.

## Deconfliction

The current version adds a first deconfliction layer.

It checks pairwise drone separation and flags when drones are too close.
This produces:
- `deconfliction_active`
- `collision_risk`
- `deconfliction_pairs`

The current task logic then blocks lower-priority drones with `hold_position_deconflict` if needed.

This is a simple first version of deconfliction, not a full collision avoidance system.

## Swarm health states

The module distinguishes between normal, degraded, and failure states.

### Degraded state

The swarm enters degraded mode when coordination is still possible but one or more issues are active.

Examples:
- stale drones
- heartbeat loss
- deconfliction active
- no target owner
- handover in progress

### Failure state

The swarm enters failure mode when coordination is no longer sufficient to continue normally.

Examples:
- too few healthy drones
- no valid parent available

These states are exposed to the Behaviour Tree through BT flags.

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
- `swarm_degraded`
- `swarm_failure`
- degraded reasons
- failure reason
- `parent_lost`
- `parent_reassigned`
- `converge_complete`
- `target_known`
- estimated target position when available
- `target_owner_id`
- `target_tracking_drone_id`
- `pending_target_owner_id`
- `handover_state`
- `target_handover_required`
- `target_handover_complete`
- `target_owner_lock_active`
- `deconfliction_active`
- `collision_risk`
- `deconfliction_pairs`

## BT integration

The interface exposes clean BT related flags.

Current BT related outputs are:
- `swarm_coordinated`
- `roles_assigned`
- `priority_list_sent`
- `role_election_failed`
- `swarm_degraded`
- `swarm_failure`
- `parent_lost`
- `parent_reassigned`
- `converge_complete`
- `target_handover_required`
- `target_handover_complete`
- `deconfliction_active`
- `collision_risk`

This means the swarm module is already producing the main coordination signals needed by the Behaviour Tree.

## Current files

`config.py`  
Stores swarm coordination settings such as role election weights, timeout values, handover rules, stability locks, and separation limits.

`models.py`  
Defines the internal drone state and swarm decision data structures.

`schemas.py`  
Defines the validated input and output message schemas.

`utils.py`  
Contains helper functions for scoring, target estimation, freshness checks, heartbeat checks, target ownership, handover candidate selection, deconfliction, task assignment, and distance checks.

`coordinator.py`  
Contains the main swarm coordination logic.

`interface.py`  
Provides a clean interface for the rest of the autonomy stack and the BT.

`adapters.py`  
Converts incoming payloads into validated message objects and internal drone state objects, and converts decisions into outbound command messages.

`main.py`  
Runs a local demo for role election, clean handover, deconfliction, and parent reassignment.

`run_swarm.py`  
Runs timestamped swarm replay input through the coordinator and logs the outputs.

`scenarios/`  
Contains replay scenario files for targeted testing of swarm behaviours.

`Tests/`  
Contains the automated MAS test suite.

`README.md`  
Explains the purpose, structure, and current status of this module.

## Replay scenarios

The replay scenario folder is intended to contain:
- `scenario_handover.jsonl`
- `scenario_parent_timeout.jsonl`
- `scenario_deconfliction.jsonl`
- `scenario_swarm_failure.jsonl`

These files can be left empty as placeholders during setup, but they should contain real newline JSON events if replay validation is meant to be fully complete.

## Test coverage

The automated test suite currently covers:
- parent selection
- handover request
- handover completion
- handover timeout
- heartbeat loss
- parent timeout and reassignment
- reassignment stability lock
- deconfliction trigger
- deconfliction clearing
- swarm failure when healthy drones are too few
- degraded mode without full failure

All current MAS tests are passing.

## Current status

The MAS module has been validated locally.

This means:
- the core coordination logic is working
- the command outputs are stable
- the BT-facing flags are stable
- the main coordination behaviours are covered by tests

At this point, the MAS module is effectively complete on the coordination logic side.

## What is still outside this module

The remaining work is outside the MAS logic itself.

This includes:
- real inter-drone message transport
- live state feeds from the rest of the autonomy stack
- execution of swarm command messages on the drones
- full end-to-end integration with BT, EKF, A*, and perception

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
- it applies stability locks to prevent churn
- it flags deconfliction risk
- it produces command-style outputs for downstream modules
- it reports clean coordination flags to the Behaviour Tree

This makes it the core swarm coordination layer for the project.