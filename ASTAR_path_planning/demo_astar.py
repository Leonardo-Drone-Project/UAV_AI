from astar import GRID_ROWS, GRID_COLS, CELL_SIZE_M, plan_path, xy_m_to_rc


DOWNSAMPLE_STEP = 2


def make_grid(rows: int, cols: int):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    wall_r = rows // 2
    for c in range(cols):
        grid[wall_r][c] = 1

    gap_c = cols // 2
    for dc in range(-2, 3):
        grid[wall_r][gap_c + dc] = 0

    for r in range(15, 35):
        grid[r][20] = 1

    for c in range(60, 80):
        grid[70][c] = 1

    return grid


def metres_path_to_cells(path_xy_m):
    return [xy_m_to_rc(x, y, CELL_SIZE_M) for (x, y) in path_xy_m]


def print_grid_downsampled(grid, path_cells, start_cell, goal_cell, step=2):
    path_set = set(path_cells) if path_cells else set()
    rows, cols = len(grid), len(grid[0])

    for r in range(0, rows, step):
        line = []
        for c in range(0, cols, step):
            if (r, c) == start_cell:
                line.append("S")
            elif (r, c) == goal_cell:
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

    start_xy_m = (10.0, 5.0)
    goal_xy_m = (80.0, 60.0)

    path_xy_m, meta = plan_path(
        start_xy_m=start_xy_m,
        goal_xy_m=goal_xy_m,
        occupancy_grid=grid,
        cell_size_m=CELL_SIZE_M,
        diagonal=True,
        simplify=True,
        line_of_sight_smoothing=True,
        waypoint_spacing_m=3.0,
        inflation_radius_cells=1,
        snap_start_goal_to_free=True,
        snap_search_radius_cells=5,
        use_obstacle_proximity_cost=True,
        obstacle_proximity_distance_cells=2,
        obstacle_proximity_gain=0.75,
        turn_penalty=0.05,
        return_metadata=True,
    )

    if path_xy_m is None:
        print("No path found.")
        print(meta)
        return

    start_cell = xy_m_to_rc(start_xy_m[0], start_xy_m[1], CELL_SIZE_M)
    goal_cell = xy_m_to_rc(goal_xy_m[0], goal_xy_m[1], CELL_SIZE_M)
    path_cells = metres_path_to_cells(path_xy_m)

    print(f"A* planning time: {meta['planning_time_ms']:.2f} ms")
    print(f"Expanded nodes: {meta['expanded_nodes']}")
    print(f"Path length: {meta['path_length_m']:.2f} m")
    print(f"Path cost: {meta['path_cost']:.2f}")
    print(f"Raw waypoint count: {meta['raw_waypoint_count']}")
    print(f"Final waypoint count: {meta['final_waypoint_count']}")
    print(f"Snapped start: {meta['snapped_start']}")
    print(f"Snapped goal: {meta['snapped_goal']}")
    print(f"Replan recommended: {meta['replan_recommended']}")
    print(f"First 8 waypoints: {path_xy_m[:8]}\n")

    print_grid_downsampled(
        grid=grid,
        path_cells=path_cells,
        start_cell=start_cell,
        goal_cell=goal_cell,
        step=DOWNSAMPLE_STEP,
    )


if __name__ == "__main__":
    main()

