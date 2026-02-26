import numpy as np
from grid.grid import *
from utils.constants import *
from utils.utils import swap_agents

# --- Order Strategies ---
def order_row(positions: np.ndarray) -> np.ndarray:
    return positions  # np.argwhere already returns row-major order

def order_random(positions: np.ndarray) -> np.ndarray:
    np.random.shuffle(positions)
    return positions


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
    direction = np.random.choice(['up', 'down', 'left', 'right'])
    for step in range(1, size):
        if direction == 'up':    ni, nj = (i - step) % size, j
        if direction == 'down':  ni, nj = (i + step) % size, j
        if direction == 'left':  ni, nj = i, (j - step) % size
        if direction == 'right': ni, nj = i, (j + step) % size
        if grid[ni, nj] == EMPTY:
            swap_agents(grid, i, j, ni, nj)
            return

def srun(grid: np.ndarray, H: int, order, move, max_iter=10_000) -> int:
    for iteration in range(1, max_iter + 1):
        mask = compute_mask(grid, H)
        unhappies = np.argwhere(~mask)
        if len(unhappies) == 0:
            return iteration
        for pos in order(unhappies):
            move(grid, pos[0], pos[1])
    print(f"Unhappy agents remaining: {np.sum(~compute_mask(grid, H) & (grid != EMPTY))}")
    return -1
