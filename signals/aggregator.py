"""Signal aggregation bus with conflict resolution."""

from __future__ import annotations

from collections import defaultdict

import structlog

from signals.base import Direction, SignalEvent

logger = structlog.get_logger(__name__)


class SignalBus:
    """Collects signals from multiple strategies and resolves conflicts."""

    def __init__(self, max_concurrent_positions: int = 15) -> None:
        self.max_concurrent = max_concurrent_positions
        self._signals: list[SignalEvent] = []
        self._open_positions: set[str] = set()

    @property
    def signals(self) -> list[SignalEvent]:
        return list(self._signals)

    def set_open_positions(self, tickers: set[str]) -> None:
        """Update the set of currently open positions."""
        self._open_positions = tickers

    def emit(self, signal: SignalEvent) -> None:
        """Add a signal to the bus."""
        self._signals.append(signal)

    def emit_batch(self, signals: list[SignalEvent]) -> None:
        """Add multiple signals to the bus."""
        self._signals.extend(signals)

    def clear(self) -> None:
        """Clear all pending signals."""
        self._signals.clear()

    def resolve(self) -> list[SignalEvent]:
        """Resolve conflicts and return filtered signals.

        Rules:
        1. Opposite directions on same ticker → suppress both
        2. Same direction on same ticker → weighted average strength
        3. Respect max concurrent position limit
        """
        if not self._signals:
            return []

        # Group by ticker
        by_ticker: dict[str, list[SignalEvent]] = defaultdict(list)
        for signal in self._signals:
            by_ticker[signal.ticker].append(signal)

        resolved: list[SignalEvent] = []

        for ticker, ticker_signals in by_ticker.items():
            longs = [s for s in ticker_signals if s.direction == Direction.LONG]
            shorts = [s for s in ticker_signals if s.direction == Direction.SHORT]

            if longs and shorts:
                # Conflict: opposite directions → suppress both
                logger.info(
                    "signal_conflict_suppressed",
                    ticker=ticker,
                    long_strategies=[s.strategy_id for s in longs],
                    short_strategies=[s.strategy_id for s in shorts],
                )
                continue

            # Same direction: merge into single signal
            active = longs or shorts
            if not active:
                continue

            merged = self._merge_signals(active)
            resolved.append(merged)

        # Sort by strength (highest priority first)
        resolved.sort(key=lambda s: s.strength, reverse=True)

        # Enforce position limit
        available_slots = self.max_concurrent - len(self._open_positions)
        if available_slots <= 0:
            logger.warning("max_positions_reached", open=len(self._open_positions))
            return []

        # Exclude tickers already in portfolio
        resolved = [s for s in resolved if s.ticker not in self._open_positions]

        # Cap to available slots
        if len(resolved) > available_slots:
            logger.info(
                "signals_capped_by_position_limit",
                total=len(resolved),
                allowed=available_slots,
            )
            resolved = resolved[:available_slots]

        return resolved

    @staticmethod
    def _merge_signals(signals: list[SignalEvent]) -> SignalEvent:
        """Merge multiple same-direction signals into one.

        Strength = weighted average. Metadata from highest-strength signal.
        """
        if len(signals) == 1:
            return signals[0]

        total_strength = sum(s.strength for s in signals)
        avg_strength = total_strength / len(signals)

        # Use the strongest signal as base
        strongest = max(signals, key=lambda s: s.strength)

        return SignalEvent(
            ticker=strongest.ticker,
            direction=strongest.direction,
            strength=min(avg_strength, 1.0),
            strategy_id="+".join(dict.fromkeys(s.strategy_id for s in signals)),
            timestamp=strongest.timestamp,
            metadata={
                **strongest.metadata,
                "merged_from": [s.strategy_id for s in signals],
                "individual_strengths": [s.strength for s in signals],
            },
        )
