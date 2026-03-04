from __future__ import annotations
import numpy as np
from grid import Grid
from strategies import OrderStrategy, MoveStrategy
from utils.constants import EMPTY


class Simulation:
    def __init__(self, grid: Grid, threshold: int,
                 order_strategy: OrderStrategy,
                 move_strategy: MoveStrategy,
                 max_iter: int = 500):
        self.grid = grid
        self.H = threshold
        self.order = order_strategy
        self.move = move_strategy
        self.max_iter = max_iter

    def run(self) -> int:
        """Run until stable or max_iter. Returns iteration count or -1."""
        for it in range(1, self.max_iter + 1):
            happy, phi = self.grid.compute_happiness()
            unhappies = np.argwhere(~happy)
            if len(unhappies) == 0:
                return it
            for r, c in self.order.order(unhappies, phi):
                self.move.move(self.grid, r, c)
        remaining = np.sum(~self.grid.compute_happiness()[0] & (self.grid.data != EMPTY))
        print(f"Unhappy agents remaining: {remaining}")
        return -1
