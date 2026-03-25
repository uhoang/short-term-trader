"""Feature 5: Online learning threshold adjustment.

Tracks rolling success rates per strategy and auto-adjusts signal thresholds
without any external ML library. Uses EWMA to weight recent trades more heavily.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from backtest.optimizer import PARAM_SPACES
from ml.utils import ML_CONFIG_DIR

logger = structlog.get_logger(__name__)

STATE_PATH = ML_CONFIG_DIR / "ml_online_threshold_state.json"

# Which params are "threshold" params that gate signal generation
# and which direction means "tighten" (make harder to trigger)
THRESHOLD_PARAMS: dict[str, dict[str, str]] = {
    "catalyst_capture": {
        "event_score_min": "increase",  # higher = stricter
        "atr_ratio_min": "increase",
    },
    "volatility_breakout": {
        "atr_ratio_min": "increase",
        "volume_spike_min": "increase",
    },
    "mean_reversion": {
        "vwap_dev_threshold": "decrease",  # more negative = stricter
        "rsi_threshold": "decrease",  # lower = stricter oversold
    },
    "sector_momentum": {
        "top_n": "decrease",  # fewer picks = stricter
        "bottom_n": "decrease",
    },
}


@dataclass
class ThresholdState:
    """Persisted state for one strategy's threshold tracker."""

    strategy_id: str
    ewma_success_rate: float = 0.5
    n_trades: int = 0
    baseline_success_rate: float = 0.5
    current_multiplier: float = 1.0  # 1.0 = no adjustment
    last_updated: str = ""


class OnlineThresholdAdjuster:
    """Tracks success rates and adjusts strategy thresholds in real time.

    How it works:
    - Maintains an EWMA of win rate per strategy
    - When EWMA drops below 80% of baseline → tighten thresholds
    - When EWMA rises above 120% of baseline → loosen thresholds
    - All adjustments are clamped to PARAM_SPACES bounds
    """

    def __init__(
        self,
        alpha: float = 0.1,
        tighten_factor: float = 1.15,
        loosen_factor: float = 0.90,
        min_trades: int = 20,
    ) -> None:
        """
        Args:
            alpha: EWMA decay factor (higher = more weight on recent trades)
            tighten_factor: Multiplier applied when tightening
            loosen_factor: Multiplier applied when loosening
            min_trades: Minimum trades before adjustments kick in
        """
        self.alpha = alpha
        self.tighten_factor = tighten_factor
        self.loosen_factor = loosen_factor
        self.min_trades = min_trades
        self._states: dict[str, ThresholdState] = {}

    def update(self, strategy_id: str, trade_profitable: bool) -> ThresholdState:
        """Update EWMA success rate after each closed trade.

        Args:
            strategy_id: Which strategy the trade came from
            trade_profitable: True if return_pct > 0

        Returns:
            Updated ThresholdState for this strategy
        """
        if strategy_id not in self._states:
            self._states[strategy_id] = ThresholdState(strategy_id=strategy_id)

        state = self._states[strategy_id]
        outcome = 1.0 if trade_profitable else 0.0

        if state.n_trades == 0:
            state.ewma_success_rate = outcome
        else:
            state.ewma_success_rate = (
                self.alpha * outcome + (1 - self.alpha) * state.ewma_success_rate
            )

        state.n_trades += 1
        state.last_updated = datetime.now().isoformat()

        # Compute multiplier based on success rate vs baseline
        if state.n_trades >= self.min_trades and state.baseline_success_rate > 0:
            ratio = state.ewma_success_rate / state.baseline_success_rate
            if ratio < 0.8:
                # Success dropping — tighten
                state.current_multiplier = self.tighten_factor
            elif ratio > 1.2:
                # Success rising — loosen
                state.current_multiplier = self.loosen_factor
            else:
                # Within normal range — no adjustment
                state.current_multiplier = 1.0

        logger.info(
            "threshold_updated",
            strategy=strategy_id,
            ewma=round(state.ewma_success_rate, 3),
            multiplier=state.current_multiplier,
            n_trades=state.n_trades,
        )
        return state

    def get_adjusted_params(self, strategy_id: str, base_params: dict[str, Any]) -> dict[str, Any]:
        """Return params with thresholds tightened/loosened based on success rate.

        Args:
            strategy_id: Which strategy to adjust
            base_params: Original strategy parameters

        Returns:
            Adjusted parameters (clamped to PARAM_SPACES bounds)
        """
        state = self._states.get(strategy_id)
        if state is None or state.current_multiplier == 1.0:
            return dict(base_params)

        adjusted = dict(base_params)
        threshold_map = THRESHOLD_PARAMS.get(strategy_id, {})
        param_bounds = PARAM_SPACES.get(strategy_id, {})
        multiplier = state.current_multiplier

        for param_name, direction in threshold_map.items():
            if param_name not in adjusted:
                continue

            value = adjusted[param_name]
            bounds = param_bounds.get(param_name, {})
            low = bounds.get("low", value * 0.5)
            high = bounds.get("high", value * 2.0)

            if direction == "increase":
                # Tighten = increase, loosen = decrease
                new_value = value * multiplier
            else:
                # Tighten = decrease, loosen = increase
                new_value = value / multiplier

            # Clamp to bounds
            if isinstance(value, int):
                new_value = int(np.clip(round(new_value), low, high))
            else:
                new_value = float(np.clip(new_value, low, high))

            adjusted[param_name] = new_value

        return adjusted

    def calibrate_baseline(self, backtest_result: dict) -> None:
        """Set baseline success rates from a backtest result.

        Args:
            backtest_result: Dict with 'trades' list, each having
                'strategy'/'strategy_id' and 'return_pct' keys
        """
        trades = backtest_result.get("trades", [])
        strategy_wins: dict[str, int] = {}
        strategy_total: dict[str, int] = {}

        for trade in trades:
            sid = trade.get("strategy", trade.get("strategy_id", "unknown"))
            # Handle concatenated strategy IDs (e.g., "mean_reversion+mean_reversion")
            sid = sid.split("+")[0] if "+" in sid else sid

            strategy_total[sid] = strategy_total.get(sid, 0) + 1
            if trade.get("return_pct", 0) > 0:
                strategy_wins[sid] = strategy_wins.get(sid, 0) + 1

        for sid, total in strategy_total.items():
            wins = strategy_wins.get(sid, 0)
            baseline = wins / total if total > 0 else 0.5

            if sid not in self._states:
                self._states[sid] = ThresholdState(strategy_id=sid)
            self._states[sid].baseline_success_rate = baseline

            logger.info(
                "baseline_calibrated",
                strategy=sid,
                baseline=round(baseline, 3),
                trades=total,
            )

    def get_state(self, strategy_id: str) -> ThresholdState | None:
        """Get current state for a strategy."""
        return self._states.get(strategy_id)

    def get_all_states(self) -> dict[str, ThresholdState]:
        """Get all strategy states."""
        return dict(self._states)

    def save_state(self, path: Path | str = STATE_PATH) -> None:
        """Save state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: asdict(state) for sid, state in self._states.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: Path | str = STATE_PATH) -> None:
        """Load state from JSON."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._states = {}
        for sid, state_dict in data.items():
            self._states[sid] = ThresholdState(**state_dict)
