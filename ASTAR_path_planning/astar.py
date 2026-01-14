# astar.py
import heapq
import math
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]          # 0 = free, 1 = obstacle
Point = Tuple[int, int]         # (row, col)


def heuristic(a: Point, b: Point, diagonal: bool) -> float:
    # Use Euclidean for diagonal grids, Manhattan for 4-neighbour grids
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
    """
    Remove unnecessary intermediate points that lie on a straight line.
    """
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
