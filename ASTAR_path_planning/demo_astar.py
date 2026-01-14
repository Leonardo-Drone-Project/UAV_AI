import time
from astar import astar, simplify_path


# -----------------------------
# Configuration 
# -----------------------------
GRID_SIZE = 100          # 100 x 100 grid
CELL_SIZE_M = 1.0        # 1 grid cell = 1 metre
DOWNSAMPLE_STEP = 4      # For terminal visualisation only


# -----------------------------
# Grid generation
# -----------------------------
def make_grid(rows: int, cols: int):
    """
    Create a simple test grid with a horizontal obstacle wall and a gap.
    """
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    wall_r = rows // 2
    for c in range(cols):
        grid[wall_r][c] = 1

    # Create a gap in the wall
    grid[wall_r][cols // 2] = 0

    return grid


# -----------------------------
# Visualisation (downsampled)
# -----------------------------
def print_grid_downsampled(grid, path, start, goal, step=4):
    """
    Print a downsampled view of the grid so large maps don't flood the terminal.
    """
    path_set = set(path) if path else set()
    rows, cols = len(grid), len(grid[0])

    for r in range(0, rows, step):
        line = []
        for c in range(0, cols, step):
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


# -----------------------------
# Coordinate conversion
# -----------------------------
def metres_to_cell(xy_m):
    """
    Convert metres -> grid cell indices.
    """
    return (int(xy_m[1] / CELL_SIZE_M), int(xy_m[0] / CELL_SIZE_M))


def cell_to_metres(cell):
    """
    Convert grid cell indices -> metres.
    """
    return (cell[1] * CELL_SIZE_M, cell[0] * CELL_SIZE_M)


# -----------------------------
# Main demo
# -----------------------------
def main():
    grid = make_grid(GRID_SIZE, GRID_SIZE)

    # Start / goal defined in METRES (physical space)
    start_xy_m = (10.0, 5.0)
    goal_xy_m = (80.0, 60.0)

    start_cell = metres_to_cell(start_xy_m)
    goal_cell = metres_to_cell(goal_xy_m)

    # Run A*
    t0 = time.perf_counter()
    path_cells = astar(grid, start_cell, goal_cell, diagonal=True)
    t1 = time.perf_counter()

    if path_cells is None:
        print("No path found.")
        return

    # Simplify path and convert to metres
    path_cells = simplify_path(path_cells)
    path_metres = [cell_to_metres(p) for p in path_cells]

    # Output summary
    print(f"A* time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Waypoints (metres): {len(path_metres)}")
    print(f"First 8 waypoints: {path_metres[:8]}\n")

    # Visualise (downsampled)
    print_grid_downsampled(
        grid,
        path_cells,
        start_cell,
        goal_cell,
        step=DOWNSAMPLE_STEP
    )


if __name__ == "__main__":
    main()


