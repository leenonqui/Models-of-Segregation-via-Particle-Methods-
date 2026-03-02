# --- Imports ---
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from utils.constants import RED, BLUE, EMPTY, SIZE, EMPTY_FRAC, DISTRIBUTION
from grid.grid import create_grid
from simulation.simulation import *
from copy import deepcopy

# --- Main ---
def main() -> None:
  # --- Part a ---
  seeds = [42, 67, 69, 121, 7]
  orders = [order_by_row, order_random, order_by_unhappiness]
  moves = [move_horizontal, move_random]
  results = {}

  for seed in seeds:
    np.random.seed(seed)
    grid = create_grid(SIZE, EMPTY_FRAC, DISTRIBUTION)
    results[seed] = {"init": deepcopy(grid)}
    for order in orders:
        for move in moves:
            g = deepcopy(grid)
            iter_count = srun(g, 4, order, move)
            results[seed][f"{order.__name__}_{move.__name__}"] = (g, iter_count)

  # --- Visualization ---
  cmap = ListedColormap(['white', 'red', 'blue'])
  for seed, res in results.items():
    n_cols = 1 + len(orders) * len(moves)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(f"Seed: {seed}")

    axes[0].imshow(res["init"], cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title("Initial")

    for idx, (order, move) in enumerate([(o, m) for o in orders for m in moves]):
      grid, iter_count = res[f"{order.__name__}_{move.__name__}"]
      title = f"{order.__name__}\n{move.__name__}\n"
      title += f"iter: {iter_count}" if iter_count != -1 else "bound hit"
      axes[idx + 1].imshow(grid, cmap=cmap, vmin=0, vmax=2)
      axes[idx + 1].set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  main()
