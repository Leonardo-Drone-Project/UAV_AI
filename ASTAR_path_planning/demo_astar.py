from astar import plan_path

GRID_SIZE = 100


def xy_to_rc(xy):
    x, y = xy
    return int(round(y)), int(round(x))


def print_grid(grid, path_xy=None, start_xy=None, goal_xy=None):
    path_cells = set()

    if path_xy:
        for xy in path_xy:
            path_cells.add(xy_to_rc(xy))

    start_rc = xy_to_rc(start_xy) if start_xy else None
    goal_rc = xy_to_rc(goal_xy) if goal_xy else None

    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[0])):
            cell = (r, c)

            if cell == start_rc:
                row += "S "
            elif cell == goal_rc:
                row += "G "
            elif cell in path_cells:
                row += "* "
            elif grid[r][c] == 1:
                row += "# "
            else:
                row += ". "
        print(row)


def build_demo_grid(rows, cols):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Vertical obstacle wall with gap
    for r in range(8, 24):
        grid[r][10] = 1
    grid[16][10] = 0
    grid[17][10] = 0

    # Horizontal obstacle wall with gap
    for c in range(0, 28):
        grid[24][c] = 1
    grid[24][15] = 0
    grid[24][16] = 0

    # Small lower obstacle
    for c in range(25, 34):
        grid[32][c] = 1

    return grid


def main():
    grid = build_demo_grid(GRID_SIZE, GRID_SIZE)

    start_xy = (10.0, 5.0)
    goal_xy = (80.0, 60.0)

    path_xy, meta = plan_path(
        start_xy_m=start_xy,
        goal_xy_m=goal_xy,
        occupancy_grid=grid,
        diagonal=True,
        simplify=False,
        line_of_sight_smoothing=False,
        waypoint_spacing_m=1.0,
        inflation_radius_cells=0,
        snap_start_goal_to_free=False,
        use_obstacle_proximity_cost=True,
        obstacle_proximity_distance_cells=2,
        obstacle_proximity_gain=0.75,
        turn_penalty=0.05,
        return_metadata=True,
    )

    print(f"A* planning time: {meta['planning_time_ms']:.2f} ms")
    print(f"Expanded nodes: {meta['expanded_nodes']}")
    print(f"Path length: {meta['path_length_m']:.2f} m")
    print(f"Path cost: {meta['path_cost']:.2f}")
    print(f"Success: {meta['success']}")
    print()

    print_grid(grid, path_xy=path_xy, start_xy=start_xy, goal_xy=goal_xy)


if __name__ == "__main__":
    main()

