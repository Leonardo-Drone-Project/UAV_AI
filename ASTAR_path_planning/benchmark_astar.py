# benchmark_astar.py
import random
import time
from statistics import mean
from astar import astar


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


def run_case(rows, cols, obstacle_prob, runs=10, diagonal=True):
    times_ms = []
    successes = 0

    start = (1, 1)
    goal = (rows - 2, cols - 2)

    for i in range(runs):
        grid = random_grid(rows, cols, obstacle_prob, seed=i)
        clear_cell(grid, start)
        clear_cell(grid, goal)

        t0 = time.perf_counter()
        path = astar(grid, start, goal, diagonal=diagonal)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000)
        if path is not None:
            successes += 1

    return mean(times_ms), max(times_ms), successes / runs


def main():
    configs = [
        (50, 50, 0.05),
        (100, 100, 0.05),
        (150, 150, 0.05),
        (100, 100, 0.10),
        (100, 100, 0.20),
    ]

    print("rows x cols | obstacle% | avg_ms | max_ms | success_rate")
    for rows, cols, p in configs:
        avg_ms, max_ms, sr = run_case(rows, cols, p, runs=10, diagonal=True)
        print(f"{rows:3d}x{cols:3d} | {p:8.2f} | {avg_ms:6.2f} | {max_ms:6.2f} | {sr:11.2f}")


if __name__ == "__main__":
    main()
