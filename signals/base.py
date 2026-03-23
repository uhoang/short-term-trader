"""Base signal classes for the strategy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class SignalEvent:
    """Represents a trading signal emitted by a strategy."""

    ticker: str
    direction: Direction
    strength: float  # 0.0 to 1.0
    strategy_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be 0-1, got {self.strength}")
