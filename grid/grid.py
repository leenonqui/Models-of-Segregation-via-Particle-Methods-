import numpy as np
from utils.constants import EMPTY, RED, BLUE

def create_grid(size: int, empty_frac: float, distribution: float) -> np.ndarray:
    n = size * size
    n_empty = int(n * empty_frac)
    n_red = int(n * (1 - empty_frac) * distribution)
    n_blue = n - n_empty - n_red
    elements = np.array([RED] * n_red + [BLUE] * n_blue + [EMPTY] * n_empty)
    np.random.shuffle(elements)
    return elements.reshape(size, size)

def compute_mask_same(grid: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    size = grid.shape[0]
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    same = np.zeros((size, size), dtype=int)
    for di, dj in offsets:
        shifted = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
        same += (shifted == grid) & (shifted != EMPTY)
    mask = same >= H
    mask[grid == EMPTY] = True
    return mask, same
