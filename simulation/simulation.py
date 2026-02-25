import numpy as np
from grid import *
from utils.constants import EMPTY
from utils.utils import swap_agents

# --- Order Strategies ---
def order_row(grid: np.ndarray) -> np.ndarray:
  size = grid.shape[0]
  return np.array([(i, j) for i in range(size) for j in range(size) if grid[i, j] != EMPTY])

def order_random(grid: np.ndarray) -> np.ndarray:
  size = grid.shape[0]
  positions = np.array([(i, j) for i in range(size) for j in range(size) if grid[i, j] != EMPTY])
  return np.random.shuffle(positions)

# --- Move Strategies ---
def move_horizontal(grid: np.ndarray, i: int, j: int):
    size = grid.shape[0]
    for step in range(1, size):
        right = (j + step) % size
        left  = (j - step) % size
        if grid[i, right] == EMPTY and grid[i, left] == EMPTY:
            swap_agents(grid, i, j, i, np.random.choice([right, left]))
            return
        if grid[i, right] == EMPTY:
            swap_agents(grid, i, j, i, right)
            return
        if grid[i, left] == EMPTY:
            swap_agents(grid, i, j, i, left)
            return

def move_random(grid: np.ndarray, i: int, j: int):
    size = grid.shape[0]
    for step in range(1, size):
        up    = (i - step) % size
        down  = (i + step) % size
        right = (j + step) % size
        left  = (j - step) % size

        candidates = []
        if grid[up,   j] == EMPTY: candidates.append((up,   j))
        if grid[down, j] == EMPTY: candidates.append((down, j))
        if grid[i, right] == EMPTY: candidates.append((i, right))
        if grid[i, left]  == EMPTY: candidates.append((i, left))

        if candidates:
            ni, nj = candidates[np.random.choice(len(candidates))]
            grid[i, j], grid[ni, nj] = grid[ni, nj], grid[i, j]
            return

def run() -> int:
    ...
