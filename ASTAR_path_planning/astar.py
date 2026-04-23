import heapq
import math
import time
from typing import Dict, List, Optional, Tuple, Any

Grid = List[List[int]]          # 0 = free, 1 = obstacle
CostGrid = List[List[float]]    # additional traversal cost per cell
Point = Tuple[int, int]         # (row, col)
XYm = Tuple[float, float]       # (x_m, y_m)

# ----------------------------
# Locked design choice
# ----------------------------
GRID_ROWS = 100
GRID_COLS = 100
CELL_SIZE_M = 1.0
COVERAGE_M = (GRID_COLS * CELL_SIZE_M, GRID_ROWS * CELL_SIZE_M)


def heuristic(a: Point, b: Point, diagonal: bool) -> float:
    if diagonal:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(grid: Grid, r: int, c: int) -> bool:
    rows, cols = len(grid), len(grid[0])
    return 0 <= r < rows and 0 <= c < cols


def _is_free(grid: Grid, r: int, c: int) -> bool:
    return _in_bounds(grid, r, c) and grid[r][c] == 0


def validate_grid(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("occupancy grid must not be empty")

    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            raise ValueError("occupancy grid must be rectangular")
        for cell in row:
            if cell not in (0, 1):
                raise ValueError("occupancy grid must contain only 0 or 1 values")


def validate_cost_grid(cost_grid: CostGrid, rows: int, cols: int) -> None:
    if len(cost_grid) != rows or len(cost_grid[0]) != cols:
        raise ValueError("cost_grid must match occupancy grid dimensions")

    for row in cost_grid:
        if len(row) != cols:
            raise ValueError("cost_grid must be rectangular")
        for value in row:
            if value < 0.0:
                raise ValueError("cost_grid values must be non-negative")


def inflate_obstacles(grid: Grid, inflation_radius_cells: int) -> Grid:
    """
    Inflate each occupied cell by a circular safety margin.
    """
    validate_grid(grid)

    if inflation_radius_cells <= 0:
        return [row[:] for row in grid]

    rows, cols = len(grid), len(grid[0])
    inflated = [row[:] for row in grid]

    offsets = []
    r2_limit = inflation_radius_cells * inflation_radius_cells
    for dr in range(-inflation_radius_cells, inflation_radius_cells + 1):
        for dc in range(-inflation_radius_cells, inflation_radius_cells + 1):
            if dr * dr + dc * dc <= r2_limit:
                offsets.append((dr, dc))

    occupied = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    for r, c in occupied:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                inflated[nr][nc] = 1

    return inflated


def obstacle_proximity_cost_grid(grid: Grid, max_distance_cells: int = 3, gain: float = 2.0) -> CostGrid:
    """
    Build a simple weighted cost map.
    Cells near obstacles get an added traversal penalty.
    Occupied cells get 0 here because they are already blocked by the occupancy grid.
    """
    validate_grid(grid)

    rows, cols = len(grid), len(grid[0])
    cost_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]

    if max_distance_cells <= 0 or gain <= 0.0:
        return cost_grid

    occupied = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not occupied:
        return cost_grid

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                continue

            min_d = float("inf")
            for orow, ocol in occupied:
                d = math.hypot(r - orow, c - ocol)
                if d < min_d:
                    min_d = d

            if min_d <= max_distance_cells:
                penalty = gain * (max_distance_cells - min_d + 1.0) / (max_distance_cells + 1.0)
                cost_grid[r][c] = penalty

    return cost_grid


def nearest_free_cell(grid: Grid, start: Point, max_radius: Optional[int] = None) -> Optional[Point]:
    """
    Search outward for the nearest free cell.
    """
    rows, cols = len(grid), len(grid[0])
    sr, sc = start

    if not (0 <= sr < rows and 0 <= sc < cols):
        return None

    if grid[sr][sc] == 0:
        return start

    if max_radius is None:
        max_radius = max(rows, cols)

    visited = set()
    queue = [(sr, sc, 0)]
    visited.add((sr, sc))

    head = 0
    while head < len(queue):
        r, c, dist = queue[head]
        head += 1

        if dist > max_radius:
            continue

        if grid[r][c] == 0:
            return (r, c)

        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return None


def neighbors(grid: Grid, node: Point, diagonal: bool) -> List[Tuple[Point, float]]:
    """
    Return valid neighbours and step costs.
    Diagonal corner-cutting is blocked.
    """
    r, c = node
    out: List[Tuple[Point, float]] = []

    cardinal_moves = [
        ((r + 1, c), 1.0),
        ((r - 1, c), 1.0),
        ((r, c + 1), 1.0),
        ((r, c - 1), 1.0),
    ]

    for (nr, nc), cost in cardinal_moves:
        if _is_free(grid, nr, nc):
            out.append(((nr, nc), cost))

    if diagonal:
        d = math.sqrt(2.0)
        diagonal_moves = [
            (r + 1, c + 1, r + 1, c, r, c + 1),
            (r + 1, c - 1, r + 1, c, r, c - 1),
            (r - 1, c + 1, r - 1, c, r, c + 1),
            (r - 1, c - 1, r - 1, c, r, c - 1),
        ]

        for nr, nc, side1_r, side1_c, side2_r, side2_c in diagonal_moves:
            if _is_free(grid, nr, nc) and _is_free(grid, side1_r, side1_c) and _is_free(grid, side2_r, side2_c):
                out.append(((nr, nc), d))

    return out


def reconstruct(came_from: Dict[Point, Point], end: Point) -> List[Point]:
    path = [end]
    cur = end
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def path_length_cells(path: List[Point]) -> float:
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        total += math.hypot(dr, dc)
    return total


def _direction(a: Point, b: Point) -> Tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    dr = 0 if dr == 0 else dr // abs(dr)
    dc = 0 if dc == 0 else dc // abs(dc)
    return dr, dc


def simplify_path(path: List[Point]) -> List[Point]:
    if not path or len(path) < 3:
        return path

    simplified = [path[0]]
    prev_dir = _direction(path[0], path[1])

    for i in range(1, len(path) - 1):
        cur_dir = _direction(path[i], path[i + 1])
        if cur_dir != prev_dir:
            simplified.append(path[i])
        prev_dir = cur_dir

    simplified.append(path[-1])
    return simplified


def _bresenham_cells(a: Point, b: Point) -> List[Point]:
    """
    Integer grid cells on a line from a to b.
    """
    x0, y0 = a[1], a[0]
    x1, y1 = b[1], b[0]

    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


def line_of_sight(grid: Grid, a: Point, b: Point) -> bool:
    """
    True if straight segment between a and b stays in free cells.
    """
    for r, c in _bresenham_cells(a, b):
        if not _is_free(grid, r, c):
            return False
    return True


def smooth_path_line_of_sight(grid: Grid, path: List[Point]) -> List[Point]:
    """
    Remove intermediate points if a direct line of sight exists.
    """
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    i = 0
    n = len(path)

    while i < n - 1:
        furthest = i + 1
        for j in range(i + 1, n):
            if line_of_sight(grid, path[i], path[j]):
                furthest = j
            else:
                break
        smoothed.append(path[furthest])
        i = furthest

    return smoothed


def resample_path_by_spacing(path_xy_m: List[XYm], spacing_m: float) -> List[XYm]:
    """
    Reduce very dense waypoint sets.
    Keep points only when cumulative distance exceeds spacing.
    Always keep first and last point.
    """
    if not path_xy_m or len(path_xy_m) < 2 or spacing_m <= 0.0:
        return path_xy_m

    out = [path_xy_m[0]]
    accum = 0.0

    for i in range(1, len(path_xy_m)):
        prev = path_xy_m[i - 1]
        cur = path_xy_m[i]
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        accum += seg

        if accum >= spacing_m:
            out.append(cur)
            accum = 0.0

    if out[-1] != path_xy_m[-1]:
        out.append(path_xy_m[-1])

    return out


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def xy_m_to_rc(x_m: float, y_m: float, cell_size_m: float = CELL_SIZE_M) -> Point:
    col = int(round(x_m / cell_size_m))
    row = int(round(y_m / cell_size_m))
    row = _clamp(row, 0, GRID_ROWS - 1)
    col = _clamp(col, 0, GRID_COLS - 1)
    return row, col


def rc_to_xy_m(row: int, col: int, cell_size_m: float = CELL_SIZE_M) -> XYm:
    x_m = col * cell_size_m
    y_m = row * cell_size_m
    return float(x_m), float(y_m)


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    diagonal: bool = True,
    cost_grid: Optional[CostGrid] = None,
    turn_penalty: float = 0.0,
) -> Tuple[Optional[List[Point]], Dict[str, Any]]:
    """
    A* on a binary occupancy grid with optional weighted costs and turn penalty.
    Returns:
      path, metadata
    """
    validate_grid(grid)

    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("start point is out of bounds")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError("goal point is out of bounds")

    if cost_grid is not None:
        validate_cost_grid(cost_grid, rows, cols)

    t0 = time.perf_counter()

    if grid[sr][sc] == 1:
        return None, {
            "success": False,
            "failure_reason": "start_blocked",
            "expanded_nodes": 0,
            "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
            "path_length_cells": 0.0,
            "path_cost": 0.0,
        }

    if grid[gr][gc] == 1:
        return None, {
            "success": False,
            "failure_reason": "goal_blocked",
            "expanded_nodes": 0,
            "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
            "path_length_cells": 0.0,
            "path_cost": 0.0,
        }

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {start: 0.0}

    open_heap: List[Tuple[float, float, Point]] = []
    heapq.heappush(open_heap, (heuristic(start, goal, diagonal), 0.0, start))

    closed_set = set()
    expanded_nodes = 0

    while open_heap:
        current_f, current_g_from_heap, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue
        if current_g_from_heap > g_score.get(current, float("inf")):
            continue

        if current == goal:
            path = reconstruct(came_from, goal)
            planning_time_ms = (time.perf_counter() - t0) * 1000.0
            return path, {
                "success": True,
                "failure_reason": None,
                "expanded_nodes": expanded_nodes,
                "planning_time_ms": planning_time_ms,
                "path_length_cells": path_length_cells(path),
                "path_cost": g_score[goal],
            }

        closed_set.add(current)
        expanded_nodes += 1

        for nxt, step_cost in neighbors(grid, current, diagonal):
            if nxt in closed_set:
                continue

            extra_cost = 0.0
            if cost_grid is not None:
                extra_cost += cost_grid[nxt[0]][nxt[1]]

            if turn_penalty > 0.0 and current in came_from:
                prev = came_from[current]
                dir1 = _direction(prev, current)
                dir2 = _direction(current, nxt)
                if dir1 != dir2:
                    extra_cost += turn_penalty

            tentative_g = g_score[current] + step_cost + extra_cost

            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal, diagonal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt))

    planning_time_ms = (time.perf_counter() - t0) * 1000.0
    return None, {
        "success": False,
        "failure_reason": "no_path",
        "expanded_nodes": expanded_nodes,
        "planning_time_ms": planning_time_ms,
        "path_length_cells": 0.0,
        "path_cost": 0.0,
    }


def plan_path(
    start_xy_m: XYm,
    goal_xy_m: XYm,
    occupancy_grid: Grid,
    cell_size_m: float = CELL_SIZE_M,
    diagonal: bool = True,
    simplify: bool = True,
    line_of_sight_smoothing: bool = True,
    waypoint_spacing_m: float = 0.0,
    inflation_radius_cells: int = 0,
    snap_start_goal_to_free: bool = False,
    snap_search_radius_cells: Optional[int] = None,
    use_obstacle_proximity_cost: bool = False,
    obstacle_proximity_distance_cells: int = 3,
    obstacle_proximity_gain: float = 2.0,
    turn_penalty: float = 0.0,
    return_metadata: bool = False,
) -> Any:
    """
    Main project-facing function.

    Returns:
      - waypoints only, if return_metadata=False
      - (waypoints, metadata), if return_metadata=True
    """
    validate_grid(occupancy_grid)

    if len(occupancy_grid) != GRID_ROWS or len(occupancy_grid[0]) != GRID_COLS:
        raise ValueError(
            f"occupancy_grid must be {GRID_ROWS}x{GRID_COLS} for the locked design choice"
        )

    working_grid = inflate_obstacles(occupancy_grid, inflation_radius_cells)

    start_rc = xy_m_to_rc(start_xy_m[0], start_xy_m[1], cell_size_m)
    goal_rc = xy_m_to_rc(goal_xy_m[0], goal_xy_m[1], cell_size_m)

    snapped_start = False
    snapped_goal = False

    if snap_start_goal_to_free:
        new_start = nearest_free_cell(working_grid, start_rc, snap_search_radius_cells)
        new_goal = nearest_free_cell(working_grid, goal_rc, snap_search_radius_cells)

        if new_start is None or new_goal is None:
            metadata = {
                "success": False,
                "failure_reason": "unable_to_snap_start_or_goal",
                "snapped_start": False,
                "snapped_goal": False,
                "expanded_nodes": 0,
                "planning_time_ms": 0.0,
                "path_length_cells": 0.0,
                "path_length_m": 0.0,
                "path_cost": 0.0,
                "raw_waypoint_count": 0,
                "final_waypoint_count": 0,
                "replan_recommended": True,
            }
            if return_metadata:
                return None, metadata
            return None

        snapped_start = (new_start != start_rc)
        snapped_goal = (new_goal != goal_rc)
        start_rc = new_start
        goal_rc = new_goal

    cost_grid = None
    if use_obstacle_proximity_cost:
        cost_grid = obstacle_proximity_cost_grid(
            working_grid,
            max_distance_cells=obstacle_proximity_distance_cells,
            gain=obstacle_proximity_gain,
        )

    path_rc, metadata = astar(
        working_grid,
        start_rc,
        goal_rc,
        diagonal=diagonal,
        cost_grid=cost_grid,
        turn_penalty=turn_penalty,
    )

    metadata["snapped_start"] = snapped_start
    metadata["snapped_goal"] = snapped_goal
    metadata["replan_recommended"] = not metadata["success"]

    if path_rc is None:
        metadata["path_length_m"] = 0.0
        metadata["raw_waypoint_count"] = 0
        metadata["final_waypoint_count"] = 0
        if return_metadata:
            return None, metadata
        return None

    raw_waypoint_count = len(path_rc)

    if simplify:
        path_rc = simplify_path(path_rc)

    if line_of_sight_smoothing:
        path_rc = smooth_path_line_of_sight(working_grid, path_rc)

    path_xy_m = [rc_to_xy_m(r, c, cell_size_m) for (r, c) in path_rc]

    if waypoint_spacing_m > 0.0:
        path_xy_m = resample_path_by_spacing(path_xy_m, waypoint_spacing_m)

    metadata["path_length_m"] = metadata["path_length_cells"] * cell_size_m
    metadata["raw_waypoint_count"] = raw_waypoint_count
    metadata["final_waypoint_count"] = len(path_xy_m)

    if return_metadata:
        return path_xy_m, metadata
    return path_xy_m
