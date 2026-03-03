import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
from utils.constants import SIZE, EMPTY_FRAC, DISTRIBUTION
from grid import Grid
from simulation import Simulation
from strategies import (
    RowOrder, RandomOrder, UnhappinessOrder,
    HorizontalMove, RandomDirectionMove, RandomJumpMove,
)

CMAP = ListedColormap(['white', 'red', 'blue'])
FIGDIR = "figures"


def save_grid(grid, title, filename):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=2)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/{filename}.png", dpi=150)
    plt.close(fig)


def part_a_i():
    seeds = [42, 67, 69, 121, 7]
    orders = [RowOrder(), RandomOrder()]
    moves = [HorizontalMove(), RandomDirectionMove()]

    for seed in seeds:
        np.random.seed(seed)
        grid = Grid(SIZE, EMPTY_FRAC, DISTRIBUTION)
        save_grid(grid.to_array(), "Initial", f"a_i_seed{seed}_init")

        for order in orders:
            for move in moves:
                g = deepcopy(grid)
                sim = Simulation(g, threshold=4, order_strategy=order, move_strategy=move)
                iters = sim.run()
                oname = order.__class__.__name__
                mname = move.__class__.__name__
                label = f"{oname} + {mname}\niter: {iters}" if iters != -1 else f"{oname} + {mname}\nbound hit"
                save_grid(g.to_array(), label, f"a_i_seed{seed}_{oname}_{mname}")
                print(f"  seed={seed} {oname:20s} {mname:20s} -> {iters}")


def part_a_ii():
    seeds = [42, 67, 69, 121, 7]
    orders = [UnhappinessOrder()]
    moves = [RandomJumpMove()]

    for seed in seeds:
        np.random.seed(seed)
        grid = Grid(SIZE, EMPTY_FRAC, DISTRIBUTION)
        save_grid(grid.to_array(), "Initial", f"a_ii_seed{seed}_init")

        for order in orders:
            for move in moves:
                g = deepcopy(grid)
                sim = Simulation(g, threshold=4, order_strategy=order, move_strategy=move)
                iters = sim.run()
                oname = order.__class__.__name__
                mname = move.__class__.__name__
                label = f"{oname} + {mname}\niter: {iters}" if iters != -1 else f"{oname} + {mname}\nbound hit"
                save_grid(g.to_array(), label, f"a_ii_seed{seed}_{oname}_{mname}")
                print(f"  seed={seed} {oname:20s} {mname:20s} -> {iters}")


def part_b():
    seeds = [42]
    order = UnhappinessOrder()
    move = RandomJumpMove()

    for H in range(1, 9):
        for seed in seeds:
            np.random.seed(seed)
            g = Grid(SIZE, EMPTY_FRAC, DISTRIBUTION)
            sim = Simulation(g, threshold=H, order_strategy=order, move_strategy=move)
            iters = sim.run()
            label = f"H={H}, iter: {iters}" if iters != -1 else f"H={H}, bound hit"
            save_grid(g.to_array(), label, f"b_H{H}_seed{seed}")
            print(f"  H={H} seed={seed} -> {iters}")


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    print("Part a(i): baseline strategies")
    part_a_i()

    print("\nPart a(ii): proposed strategies")
    part_a_ii()

    print("\nPart b: H sweep from 1 to 8")
    part_b()
