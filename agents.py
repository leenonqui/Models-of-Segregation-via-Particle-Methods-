from __future__ import annotations
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base agent type. Subclass to define new agent kinds.

    The grid stores agents as int IDs (see kind property).
    This class exists for extensibility — e.g. custom happiness
    rules, movement preferences, or per-agent state.
    """

    @property
    @abstractmethod
    def kind(self) -> int:
        """Unique nonzero int identifying this agent type on the grid."""
        ...


class RedAgent(Agent):
    @property
    def kind(self) -> int:
        return 1


class BlueAgent(Agent):
    @property
    def kind(self) -> int:
        return 2
