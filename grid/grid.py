import numpy as np
from utils.constants import RED, BLUE, EMPTY

def create_grid(size: int, empty_frac: float, distribution: float) -> np.ndarray:
  """
  Creates a SIZExSIZE grid with randomized cell values and empty_frac empty cells
  """
  n = size*size
  n_empty = int(n * empty_frac)
  n_red = int(n * (1 - empty_frac) * distribution)
  n_blue = int(n - (n_empty + n_red))

  elements = np.array([RED]*n_red + [BLUE]*n_blue + [EMPTY]*n_empty)
  np.random.shuffle(elements)
  return elements.reshape(size, size)

def get_neighbor(grid: np.ndarray, i: int, j: int) -> np.ndarray:
  """
  returns an array of the values of the neighbors comperad to the grid point (i, j) with periodic boundary conditions
  """
  array = np.zeros(8, dtype=int)
  size = grid.shape[0]

  offsets = [(-1,-1), (-1, 0), (-1, 1),
             ( 0,-1),          ( 0, 1),
             ( 1,-1), ( 1, 0), ( 1, 1)]
  for index, offset in enumerate(offsets):
    di, dj = offset
    array[index] = grid[(i + di) % size, (j + dj) % size]

  return array


def is_happy(grid: np.ndarray, i: int, j: int, H: int) -> bool:
  """
  returns True if at leas H neihbors are of same type as the one in grid point (i,j) else false
  """
  agent = grid[i, j]
  if agent == EMPTY:
    return True

  neighbors = get_neighbor(grid, i, j)
  occupied = neighbors[neighbors != EMPTY]
  n_same_type = np.sum(occupied == agent)
  return n_same_type >= H

def compute_mask(grid: np.ndarray, H: int) -> np.ndarray:
  """
  Returns a boolean grid where True means the agent is happy
  """
  size = grid.shape[0]
  offsets = [(-1,-1), (-1, 0), (-1, 1),
              ( 0,-1),          ( 0, 1),
              ( 1,-1), ( 1, 0), ( 1, 1)]

  same = np.zeros((size, size), dtype=int)
  for di, dj in offsets:
      shifted = np.roll(np.roll(grid, di, axis=0), dj, axis=1) # we check for all cells for each possible offsett if it is same type as agent in cell i, j
      same += (shifted == grid) & (shifted != EMPTY)

  mask = same >= H
  mask[grid == EMPTY] = True
  return mask


