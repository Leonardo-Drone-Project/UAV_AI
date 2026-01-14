import heapq
import math
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]          # 0 = free, 1 = obstacle
Point = Tuple[int, int]         # (row, col)

# ----------------------------
# Locked design choice (project)
# ----------------------------
GRID_ROWS = 100
GRID_COLS = 100
CELL_SIZE_M = 1.0  # 1 grid cell = 1 m
COVERAGE_M = (GRID_COLS * CELL_SIZE_M, GRID_ROWS * CELL_SIZE_M)  # (width_m, height_m)


def heuristic(a: Point, b: Point, diagonal: bool) -> float:
    # Euclidean for diagonal, Manhattan for 4-neighbour
    if diagonal:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(grid: Grid, node: Point, diagonal: bool) -> List[Tuple[Point, float]]:
    r, c = node
    rows, cols = len(grid), len(grid[0])

    moves = [((r + 1, c), 1.0), ((r - 1, c), 1.0), ((r, c + 1), 1.0), ((r, c - 1), 1.0)]
    if diagonal:
        d = math.sqrt(2)
        moves += [
            ((r + 1, c + 1), d), ((r + 1, c - 1), d),
            ((r - 1, c + 1), d), ((r - 1, c - 1), d),
        ]

    out = []
    for (nr, nc), cost in moves:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            out.append(((nr, nc), cost))
    return out


def reconstruct(came_from: Dict[Point, Point], end: Point) -> List[Point]:
    path = [end]
    cur = end
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar(grid: Grid, start: Point, goal: Point, diagonal: bool = True) -> Optional[List[Point]]:
    """
    A* on a binary occupancy grid.
    Returns: list of (row,col) from start->goal, or None if no path.
    """
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None

    open_heap: List[Tuple[float, Point]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Point, Point] = {}
    g: Dict[Point, float] = {start: 0.0}
    in_open = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        in_open.discard(current)

        if current == goal:
            return reconstruct(came_from, goal)

        for nxt, step_cost in neighbors(grid, current, diagonal):
            tentative = g[current] + step_cost
            if nxt not in g or tentative < g[nxt]:
                came_from[nxt] = current
                g[nxt] = tentative
                f = tentative + heuristic(nxt, goal, diagonal)
                if nxt not in in_open:
                    heapq.heappush(open_heap, (f, nxt))
                    in_open.add(nxt)

    return None


def simplify_path(path: List[Point]) -> List[Point]:
    """Remove unnecessary intermediate points that lie on a straight line."""
    if not path or len(path) < 3:
        return path

    simplified = [path[0]]

    def direction(a: Point, b: Point) -> Tuple[int, int]:
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        return (0 if dr == 0 else dr // abs(dr), 0 if dc == 0 else dc // abs(dc))

    prev_dir = direction(path[0], path[1])
    for i in range(1, len(path) - 1):
        cur_dir = direction(path[i], path[i + 1])
        if cur_dir != prev_dir:
            simplified.append(path[i])
        prev_dir = cur_dir

    simplified.append(path[-1])
    return simplified


# ----------------------------
# Metres-based interface
# ----------------------------
XYm = Tuple[float, float]  # (x_m, y_m)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def xy_m_to_rc(x_m: float, y_m: float, cell_size_m: float = CELL_SIZE_M) -> Point:
    """
    Convert metres -> grid indices.
    Convention used here:
      - x_m increases to the right (cols)
      - y_m increases downward (rows)  (image/grid convention)
    """
    col = int(round(x_m / cell_size_m))
    row = int(round(y_m / cell_size_m))
    row = _clamp(row, 0, GRID_ROWS - 1)
    col = _clamp(col, 0, GRID_COLS - 1)
    return (row, col)


def rc_to_xy_m(row: int, col: int, cell_size_m: float = CELL_SIZE_M) -> XYm:
    """Convert grid indices -> metres (cell centre)."""
    x_m = col * cell_size_m
    y_m = row * cell_size_m
    return (float(x_m), float(y_m))


def plan_path(
    start_xy_m: XYm,
    goal_xy_m: XYm,
    occupancy_grid: Grid,
    cell_size_m: float = CELL_SIZE_M,
    diagonal: bool = True,
    simplify: bool = True,
) -> Optional[List[XYm]]:
    """
    Main project-facing function (locked design choice):
      - occupancy_grid must be 100x100
      - 1 cell = 1 m  -> 100m x 100m coverage

    Returns:
      List of (x_m, y_m) waypoints, or None if no path.
    """
    if len(occupancy_grid) != GRID_ROWS or len(occupancy_grid[0]) != GRID_COLS:
        raise ValueError(f"occupancy_grid must be {GRID_ROWS}x{GRID_COLS} for the locked design choice.")

    start_rc = xy_m_to_rc(start_xy_m[0], start_xy_m[1], cell_size_m)
    goal_rc = xy_m_to_rc(goal_xy_m[0], goal_xy_m[1], cell_size_m)

    path_rc = astar(occupancy_grid, start_rc, goal_rc, diagonal=diagonal)
    if path_rc is None:
        return None

    if simplify:
        path_rc = simplify_path(path_rc)

    return [rc_to_xy_m(r, c, cell_size_m) for (r, c) in path_rc]
