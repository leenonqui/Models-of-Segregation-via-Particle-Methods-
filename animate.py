"""Standalone animation of Schelling segregation. Copy-paste ready, no dependency on simulation.py."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from grid import Grid
from strategies import OrderStrategy, MoveStrategy, RandomJumpMove, RandomOrder
from utils.constants import SIZE, EMPTY_FRAC, DISTRIBUTION, EMPTY

CMAP = ListedColormap(['white', 'red', 'blue'])


def capture_run(grid: Grid, H: int, order: OrderStrategy, move: MoveStrategy,
                max_iter: int = 500) -> list[tuple[int, np.ndarray, int]]:
    """Run simulation, return list of (iteration, grid_snapshot, n_unhappy)."""
    from grid import OFFSETS
    frames = []
    for it in range(1, max_iter + 1):
        phi = np.zeros_like(grid.data, dtype=int)
        for dr, dc in OFFSETS:
            shifted = np.roll(np.roll(grid.data, dr, axis=0), dc, axis=1)
            phi += (shifted == grid.data) & (shifted != EMPTY)
        happy = phi >= H
        happy[grid.data == EMPTY] = True
        unhappies = np.argwhere(~happy)

        frames.append((it, grid.to_array().copy(), len(unhappies)))

        if len(unhappies) == 0:
            break
        for r, c in order.order(unhappies, phi):
            move.move(grid, r, c)

    return frames


def animate(frames: list[tuple[int, np.ndarray, int]], title: str = "",
            fps: int = 10, save_path: str = "schelling.mp4"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0][1], cmap=CMAP, vmin=0, vmax=2)
    txt = ax.set_title("", fontsize=12)
    ax.axis('off')

    def update(frame_idx):
        it, snap, n_unhappy = frames[frame_idx]
        im.set_data(snap)
        label = f"{title}  |  iter {it}  |  unhappy: {n_unhappy}"
        ax.set_title(label, fontsize=12)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    seed = 42
    H = 4
    np.random.seed(seed)
    grid = Grid(SIZE, EMPTY_FRAC, DISTRIBUTION)

    order = RandomOrder()
    move = RandomJumpMove()

    frames = capture_run(deepcopy(grid), H, order, move)
    animate(frames, title=f"seed={seed} H={H}", save_path="schelling_nearhappy_jump.mp4")
