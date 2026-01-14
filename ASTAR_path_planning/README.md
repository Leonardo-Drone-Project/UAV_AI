A* is an optimal path-planning algorithm that finds the shortest collision-free path by balancing travelled cost with an estimated distance to the goal. In this project, it is applied to a grid-based environment and demonstrates fast, reliable performance suitable for real-time UAV path planning in open-field scenarios.

## Benchmark Results

The table below summarises the A* planning performance across different map sizes and obstacle densities.

| Grid Size | Obstacle Density | Avg Time (ms) | Max Time (ms) | Success Rate |
|----------:|------------------:|--------------:|--------------:|-------------:|
| 50 × 50   | 0.05              | 0.35          | 0.62          | 1.00         |
| 100 × 100 | 0.05              | 1.32          | 2.29          | 1.00         |
| 150 × 150 | 0.05              | 3.05          | 5.60          | 1.00         |
| 100 × 100 | 0.10              | 2.24          | 3.63          | 1.00         |
| 100 × 100 | 0.20              | 5.59          | 7.18          | 1.00         |

These results show that A* scales predictably with grid size and obstacle density while maintaining a 100% success rate, making it suitable for onboard UAV path planning.
