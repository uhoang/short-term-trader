"""Tests for CMA-ES joint optimizer."""

from __future__ import annotations

import numpy as np

from backtest.optimizer import PARAM_SPACES
from ml.cma_joint_optimizer import REGIME_ORDER, STRATEGY_ORDER, ParameterCodec
from signals.regime import DEFAULT_REGIME_WEIGHTS


class TestParameterCodec:
    """Test encode/decode round-trip."""

    def setup_method(self):
        self.codec = ParameterCodec(
            param_spaces=PARAM_SPACES,
            regime_order=REGIME_ORDER,
            strategy_order=STRATEGY_ORDER,
        )

    def test_dimension(self):
        # Count expected params
        n_strategy = sum(len(space) for space in PARAM_SPACES.values())
        n_regime = len(REGIME_ORDER) * len(STRATEGY_ORDER)
        assert self.codec.dimension == n_strategy + n_regime

    def test_bounds_shape(self):
        lows, highs = self.codec.bounds
        assert len(lows) == self.codec.dimension
        assert len(highs) == self.codec.dimension
        assert np.all(lows <= highs)

    def test_encode_decode_round_trip(self):
        from signals.config_loader import get_defaults

        strategy_params = get_defaults()
        regime_weights = DEFAULT_REGIME_WEIGHTS

        vector = self.codec.encode(strategy_params, regime_weights)
        assert len(vector) == self.codec.dimension

        decoded_sp, decoded_rw = self.codec.decode(vector)

        # Check strategy params preserved
        for sid in STRATEGY_ORDER:
            space = PARAM_SPACES.get(sid, {})
            for pname in space:
                if pname in strategy_params.get(sid, {}):
                    orig = strategy_params[sid][pname]
                    decoded = decoded_sp[sid][pname]
                    assert (
                        abs(float(orig) - float(decoded)) < 0.01
                    ), f"{sid}.{pname}: {orig} != {decoded}"

        # Check regime weights preserved
        for regime in REGIME_ORDER:
            for sid in STRATEGY_ORDER:
                orig = regime_weights[regime][sid]
                decoded = decoded_rw[regime][sid]
                assert abs(orig - decoded) < 0.01

    def test_decode_clamps_to_bounds(self):
        # Vector with out-of-bounds values
        vector = np.ones(self.codec.dimension) * 999.0
        decoded_sp, decoded_rw = self.codec.decode(vector)

        # All regime weights should be clamped to 2.0
        for regime in REGIME_ORDER:
            for sid in STRATEGY_ORDER:
                assert decoded_rw[regime][sid] <= 2.0

    def test_param_names(self):
        names = self.codec.param_names
        assert len(names) == self.codec.dimension
        assert all(":" in n for n in names)  # All have section prefix
