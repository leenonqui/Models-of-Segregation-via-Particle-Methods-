from __future__ import annotations
import numpy as np
from utils.constants import EMPTY, RED, BLUE

OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0, 1),
           (1, -1),  (1, 0),  (1, 1)]


class Grid:
    """Periodic boundary grid backed by ndarray for vectorized operations."""

    def __init__(self, size: int, empty_frac: float, distribution: float):
        self.size = size
        n = size * size
        n_empty = int(n * empty_frac)
        n_red = int(n * (1 - empty_frac) * distribution)
        n_blue = n - n_empty - n_red
        cells = np.array([RED] * n_red + [BLUE] * n_blue + [EMPTY] * n_empty)
        np.random.shuffle(cells)
        self.data = cells.reshape(size, size)

    def swap(self, r1: int, c1: int, r2: int, c2: int):
        self.data[r1, c1], self.data[r2, c2] = self.data[r2, c2], self.data[r1, c1]

    def compute_happiness(self, H: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (happy_mask, phi) via vectorized roll."""
        phi = np.zeros_like(self.data, dtype=int)
        for dr, dc in OFFSETS:
            shifted = np.roll(np.roll(self.data, dr, axis=0), dc, axis=1)
            phi += (shifted == self.data) & (shifted != EMPTY)
        happy = phi >= H
        happy[self.data == EMPTY] = True
        return happy, phi

    def to_array(self) -> np.ndarray:
        return self.data

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Return coordinates of all 8+1 cells in the 3x3 block around (r, c)."""
        return [((r + dr) % self.size, (c + dc) % self.size) for dr in range(-1, 2) for dc in range(-1, 2)]

    def empty_neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Empty cells in the 3x3 block around (r, c)."""
        return [(nr, nc) for nr, nc in self.neighbors(r, c) if self.data[nr, nc] == EMPTY]

    def phi_at(self, r: int, c: int, agent_type: int) -> int:
        return sum(1 for nr, nc in self.neighbors(r, c) if (nr, nc) != (r, c) and self.data[nr, nc] == agent_type)
