# demo_astar.py
import time
from astar import astar, simplify_path


def make_grid(rows: int, cols: int):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Example obstacle: wall with a gap
    wall_r = rows // 2
    for c in range(cols):
        grid[wall_r][c] = 1
    grid[wall_r][cols // 2] = 0  # gap

    return grid


def print_grid(grid, path, start, goal):
    path_set = set(path) if path else set()
    for r in range(len(grid)):
        line = []
        for c in range(len(grid[0])):
            if (r, c) == start:
                line.append("S")
            elif (r, c) == goal:
                line.append("G")
            elif (r, c) in path_set:
                line.append("*")
            elif grid[r][c] == 1:
                line.append("#")
            else:
                line.append(".")
        print(" ".join(line))


def main():
    rows, cols = 25, 45
    grid = make_grid(rows, cols)

    start = (2, 2)
    goal = (rows - 3, cols - 3)

    t0 = time.perf_counter()
    path = astar(grid, start, goal, diagonal=True)
    t1 = time.perf_counter()

    if path is None:
        print("No path found.")
        return

    path_simplified = simplify_path(path)

    print(f"A* time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Raw path nodes: {len(path)} | Simplified waypoints: {len(path_simplified)}\n")
    print_grid(grid, path, start, goal)


if __name__ == "__main__":
    main()
