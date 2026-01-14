A* is an optimal path-planning algorithm that finds the shortest collision-free path by balancing travelled cost with an estimated distance to the goal. In this project, it is applied to a grid-based environment and demonstrates fast, reliable performance suitable for real-time UAV path planning in open-field scenarios.

Benchmark output: 
rows x cols | obstacle% | avg_ms | max_ms | success_rate
 50x 50 |     0.05 |   0.35 |   0.59 |        1.00
100x100 |     0.05 |   1.39 |   2.40 |        1.00
150x150 |     0.05 |   3.17 |   5.81 |        1.00
100x100 |     0.10 |   2.27 |   3.74 |        1.00
100x100 |     0.20 |   5.46 |   8.21 |        1.00