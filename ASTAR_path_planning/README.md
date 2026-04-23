# A* Path Planning

A* is used here as a grid-based UAV path planner. The module generates collision-free paths from a start point to a goal point and returns metre-based waypoints suitable for downstream navigation.

## Locked Design Choice

The planner uses these fixed project assumptions:

- Grid size: 100 x 100 cells
- Cell size: 1 metre per cell
- Total map coverage: 100 m x 100 m
- Occupancy grid values:
  - `0` = free
  - `1` = obstacle

## Current Enhancements Added

The planner now includes:

- 4-neighbour or 8-neighbour planning
- diagonal corner-cutting protection
- obstacle inflation
- optional weighted cost map support
- blocked start and goal snapping to nearest free cell
- turn penalty support for smoother path preference
- straight-line simplification
- line-of-sight path smoothing
- waypoint spacing control
- metadata output for debugging and validation
- metre-based project-facing interface through `plan_path(...)`

## Files

- `astar.py`  
  Core A* planner and project-facing planning interface.

- `benchmark_astar.py`  
  Benchmark and validation script.

- `demo_astar.py`  
  Example scenario showing path generation and planner metadata.

## Planner Outputs

When metadata is requested, the planner returns:

- success or failure
- failure reason
- planning time
- expanded nodes
- path length
- path cost
- raw waypoint count
- final waypoint count
- snapped start or goal flags
- replan recommendation flag

## Coordinate Convention

The module uses:

- `x_m` increasing to the right
- `y_m` increasing downward
- grid row = `y`
- grid column = `x`


