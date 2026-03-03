from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from grid import Grid
from utils.constants import EMPTY


# ── Ordering strategies ──────────────────────────────────────────

class OrderStrategy(ABC):
    @abstractmethod
    def order(self, unhappies: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Receive Nx2 array of (row, col) positions and phi field. Return reordered."""
        ...


class RowOrder(OrderStrategy):
    def order(self, unhappies, phi):
        return unhappies


class RandomOrder(OrderStrategy):
    def order(self, unhappies, phi):
        np.random.shuffle(unhappies)
        return unhappies


class UnhappinessOrder(OrderStrategy):
    """Most unhappy first (fewest same-type neighbors)."""
    def order(self, unhappies, phi):
        scores = phi[unhappies[:, 0], unhappies[:, 1]]
        return unhappies[np.argsort(scores)]


# ── Move strategies ──────────────────────────────────────────────

class MoveStrategy(ABC):
    @abstractmethod
    def move(self, grid: Grid, r: int, c: int) -> None:
        ...


class HorizontalMove(MoveStrategy):
    def move(self, grid: Grid, r: int, c: int):
        for step in range(1, grid.size):
            right = (c + step) % grid.size
            left = (c - step) % grid.size
            r_empty = grid.data[r, right] == EMPTY
            l_empty = grid.data[r, left] == EMPTY
            if r_empty and l_empty:
                grid.swap(r, c, r, np.random.choice([right, left]))
                return
            if r_empty:
                grid.swap(r, c, r, right)
                return
            if l_empty:
                grid.swap(r, c, r, left)
                return


class RandomDirectionMove(MoveStrategy):
    DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def move(self, grid: Grid, r: int, c: int):
        dr, dc = self.DIRS[np.random.choice(list(self.DIRS))]
        for step in range(1, grid.size):
            nr = (r + dr * step) % grid.size
            nc = (c + dc * step) % grid.size
            if grid.data[nr, nc] == EMPTY:
                grid.swap(r, c, nr, nc)
                return
class RandomJumpMove(MoveStrategy):

    def __init__(self, alpha: float = 1.5, max_retries: int = 20):
        self.alpha = alpha
        self.max_retries = max_retries

    def _sample_distance(self, size: int) -> int:
        max_d = size // 2
        distances = np.arange(1, max_d + 1)
        weights = distances.astype(float) ** (-self.alpha)
        weights /= weights.sum()
        return np.random.choice(distances, p=weights)

    def move(self, grid: Grid, r: int, c: int):
        s = grid.size
        agent_type = grid.data[r, c]
        for _ in range(self.max_retries):
            d = self._sample_distance(s)
            angle = np.random.uniform(0, 2 * np.pi)
            dr = int(round(d * np.sin(angle)))
            dc = int(round(d * np.cos(angle)))
            cr, cc = (r + dr) % s, (c + dc) % s

            empties = [(nr, nc) for nr, nc in grid.neighbors(cr, cc)
                       if grid.data[nr, nc] == EMPTY]
            if not empties:
                continue

            best = max(empties, key=lambda p: grid.phi_at(p[0], p[1], agent_type))
            grid.swap(r, c, best[0], best[1])
            return
