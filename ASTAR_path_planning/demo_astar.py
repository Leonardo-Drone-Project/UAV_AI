import time
from astar import plan_path, xy_m_to_rc, GRID_ROWS, GRID_COLS


def make_grid(rows: int, cols: int):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Wall with a gap (for sanity check)
    wall_r = rows // 2
    for c in range(cols):
        grid[wall_r][c] = 1
    grid[wall_r][cols // 2] = 0  # gap

    return grid


def print_grid(grid, path_rc, start_rc, goal_rc):
    path_set = set(path_rc) if path_rc else set()
    for r in range(len(grid)):
        line = []
        for c in range(len(grid[0])):
            if (r, c) == start_rc:
                line.append("S")
            elif (r, c) == goal_rc:
                line.append("G")
            elif (r, c) in path_set:
                line.append("*")
            elif grid[r][c] == 1:
                line.append("#")
            else:
                line.append(".")
        print(" ".join(line))


def main():
    grid = make_grid(GRID_ROWS, GRID_COLS)

    # Metres inputs (locked design: 100x100m area, 1m/cell)
    start_xy_m = (10.0, 5.0)
    goal_xy_m = (80.0, 60.0)

    t0 = time.perf_counter()
    path_xy = plan_path(start_xy_m, goal_xy_m, grid, cell_size_m=1.0, diagonal=True, simplify=True)
    t1 = time.perf_counter()

    if path_xy is None:
        print("No path found.")
        return

    # Convert the returned metre-waypoints back to grid cells purely for ASCII plotting
    path_rc = [xy_m_to_rc(x, y, 1.0) for (x, y) in path_xy]
    start_rc = xy_m_to_rc(*start_xy_m, 1.0)
    goal_rc = xy_m_to_rc(*goal_xy_m, 1.0)

    print(f"A* time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Waypoints (metres): {len(path_xy)}")
    print("First 8 waypoints:", path_xy[:8], "\n")

    print_grid(grid, path_rc, start_rc, goal_rc)


if __name__ == "__main__":
    main()


