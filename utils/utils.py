def swap_agents(grid, i1: int, j1: int, i2: int, j2: int):
  grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
