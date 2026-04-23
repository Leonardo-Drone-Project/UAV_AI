import random
from statistics import mean
from astar import plan_path


GRID_SIZE = 100


def random_grid(rows: int, cols: int, obstacle_prob: float, seed: int = 0):
    rng = random.Random(seed)
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if rng.random() < obstacle_prob:
                grid[r][c] = 1
    return grid


def clear_cell(grid, p):
    grid[p[0]][p[1]] = 0


def xy_to_rc(xy):
    x, y = xy
    return int(round(y)), int(round(x))


def path_is_valid(grid, path_xy):
    if path_xy is None or len(path_xy) == 0:
        return False

    rows, cols = len(grid), len(grid[0])

    for xy in path_xy:
        r, c = xy_to_rc(xy)
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r][c] == 1:
            return False

    return True


def run_case(obstacle_prob, runs=10):
    planning_times = []
    successes = 0
    valid_paths = 0
    path_lengths = []
    final_waypoints = []
    expanded_nodes = []

    rows = GRID_SIZE
    cols = GRID_SIZE

    start_xy = (1.0, 1.0)
    goal_xy = (cols - 2.0, rows - 2.0)

    for i in range(runs):
        grid = random_grid(rows, cols, obstacle_prob, seed=i)
        clear_cell(grid, (1, 1))
        clear_cell(grid, (rows - 2, cols - 2))

        path_xy, meta = plan_path(
            start_xy_m=start_xy,
            goal_xy_m=goal_xy,
            occupancy_grid=grid,
            diagonal=True,
            simplify=True,
            line_of_sight_smoothing=True,
            waypoint_spacing_m=2.5,
            inflation_radius_cells=0,
            snap_start_goal_to_free=False,
            use_obstacle_proximity_cost=True,
            obstacle_proximity_distance_cells=2,
            obstacle_proximity_gain=0.75,
            turn_penalty=0.05,
            return_metadata=True,
        )

        planning_times.append(meta["planning_time_ms"])
        expanded_nodes.append(meta["expanded_nodes"])

        if meta["success"]:
            successes += 1
            if (
                path_xy is not None
                and path_xy[0] == start_xy
                and path_xy[-1] == goal_xy
                and path_is_valid(grid, path_xy)
            ):
                valid_paths += 1

            path_lengths.append(meta["path_length_m"])
            final_waypoints.append(meta["final_waypoint_count"])

    return {
        "avg_ms": mean(planning_times),
        "max_ms": max(planning_times),
        "success_rate": successes / runs,
        "valid_rate": valid_paths / runs,
        "avg_len_m": mean(path_lengths) if path_lengths else 0.0,
        "avg_waypoints": mean(final_waypoints) if final_waypoints else 0.0,
        "avg_expanded_nodes": mean(expanded_nodes) if expanded_nodes else 0.0,
    }


def run_blocked_case():
    rows, cols = GRID_SIZE, GRID_SIZE
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    wall_r = rows // 2
    for c in range(cols):
        grid[wall_r][c] = 1

    start_xy = (10.0, 10.0)
    goal_xy = (90.0, 90.0)

    path_xy, meta = plan_path(
        start_xy_m=start_xy,
        goal_xy_m=goal_xy,
        occupancy_grid=grid,
        diagonal=True,
        simplify=True,
        line_of_sight_smoothing=True,
        waypoint_spacing_m=2.5,
        inflation_radius_cells=0,
        snap_start_goal_to_free=False,
        use_obstacle_proximity_cost=True,
        obstacle_proximity_distance_cells=2,
        obstacle_proximity_gain=0.75,
        turn_penalty=0.05,
        return_metadata=True,
    )
    return path_xy is None, meta["failure_reason"]


def main():
    obstacle_configs = [0.05, 0.10, 0.15, 0.20]

    print(
        "grid | obstacle% | avg_ms | max_ms | success_rate | valid_rate | avg_len_m | avg_waypoints | avg_expanded"
    )
    for p in obstacle_configs:
        result = run_case(p, runs=10)
        print(
            f"{GRID_SIZE}x{GRID_SIZE} | "
            f"{p:8.2f} | "
            f"{result['avg_ms']:6.2f} | "
            f"{result['max_ms']:6.2f} | "
            f"{result['success_rate']:12.2f} | "
            f"{result['valid_rate']:10.2f} | "
            f"{result['avg_len_m']:9.2f} | "
            f"{result['avg_waypoints']:13.2f} | "
            f"{result['avg_expanded_nodes']:12.2f}"
        )

    blocked_ok, failure_reason = run_blocked_case()
    print("\nBlocked-map validation:")
    print(f"No-path case returned None: {blocked_ok}")
    print(f"Failure reason: {failure_reason}")


if __name__ == "__main__":
    main()
