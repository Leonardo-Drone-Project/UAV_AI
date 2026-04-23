from astar import astar

# ----------------------------
# Visualisation (no libraries)
# ----------------------------
def print_grid(grid, path=None, start=None, goal=None):
    path_set = set(path) if path else set()

    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[0])):
            if (r, c) == start:
                row += "S "
            elif (r, c) == goal:
                row += "G "
            elif (r, c) in path_set:
                row += "* "
            elif grid[r][c] == 1:
                row += "# "
            else:
                row += ". "
        print(row)


# ----------------------------
# Validation helpers
# ----------------------------

# Checks if a node is within the grid bounds and not an obstacle.
def is_inside_grid(grid, node):
    r, c = node
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def is_valid_path(grid, path):
    if not path:
        return True

    for r, c in path:
        if not is_inside_grid(grid, (r, c)):
            return False
        if grid[r][c] == 1:
            return False
    return True

# Cheks if each step moves one cell at a time.
def is_connected(path):
    if not path:
        return True

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        dr = abs(r1 - r2)
        dc = abs(c1 - c2)

        # allow 4 or 8-connected movement
        if max(dr, dc) > 1:
            return False
    return True


# counts path length
def path_length(path):
    return 0 if not path else len(path) - 1

# Measures manhattan distance. 
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------
# Deterministic tests
# ----------------------------

# This checks if the program finds a path, heads for it directly and doesnt do anything crazy.
def test_straight_line():
    grid = [[0]*10 for _ in range(10)]
    start = (0, 0)
    goal = (0, 5)

    path = astar(grid, start, goal, diagonal=False)

    assert path is not None
    assert is_valid_path(grid, path)
    assert is_connected(path)

    print("[PASS] Straight line test")

# This checks if the program can find a path around an obstacle.
def test_obstacle_avoidance():
    grid = [[0]*10 for _ in range(10)]

    for i in range(1, 9):
        grid[5][i] = 1

    start = (0, 5)
    goal = (9, 5)

    path = astar(grid, start, goal, diagonal=False)

    assert path is not None
    assert is_valid_path(grid, path)
    assert is_connected(path)

    print("[PASS] Obstacle avoidance test")


# This checks if the program correctly returns None when no path exists.
def test_no_path():
    grid = [[0]*10 for _ in range(10)]

    for i in range(10):
        grid[5][i] = 1

    start = (0, 5)
    goal = (9, 5)

    path = astar(grid, start, goal, diagonal=False)

    assert path is None

    print("[PASS] No path test")


# ----------------------------
# Random stress test
# ----------------------------
import random

# Creates randim grid with given size and obstacle probability for testing.
def random_grid(n=20, obstacle_prob=0.25):
    return [
        [1 if random.random() < obstacle_prob else 0 for _ in range(n)]
        for _ in range(n)
    ]


# Runs the test multiple times on random grids to see if it works all the time.
def stress_test(trials=50):
    failures = 0

    for i in range(trials):
        grid = random_grid()

        start = (0, 0)
        goal = (len(grid)-1, len(grid[0])-1)

        path = astar(grid, start, goal, diagonal=True)

        if path is None:
            continue

        if not is_valid_path(grid, path):
            print("[FAIL] Path goes through obstacle")
            failures += 1
            continue

        if not is_connected(path):
            print("[FAIL] Path not connected")
            failures += 1
            continue

        # sanity check: path should not be absurdly long
        if path_length(path) > 4 * manhattan(start, goal):
            print("[WARN] Path seems inefficient")

    print(f"[STRESS TEST COMPLETE] Failures: {failures}/{trials}")


# ----------------------------
# Main runner
# ----------------------------
def main():
    print("Running deterministic tests...\n")

    test_straight_line()
    test_obstacle_avoidance()
    test_no_path()

    print("\nRunning stress tests...\n")
    stress_test()

    print("\nAll tests complete.")

    grid = [[0]*20 for _ in range(20)]

    # obstacle wall
    for i in range(5, 15):
        grid[10][i] = 1
        grid[i][10] = 1

    start = (0, 0)
    goal = (14, 12)

    path = astar(grid, start, goal, diagonal=True)

    print("Path length:", len(path) if path else None)
    print_grid(grid, path, start, goal)

if __name__ == "__main__":
    main()