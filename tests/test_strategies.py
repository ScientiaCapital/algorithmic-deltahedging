"""
Unit tests for options strategies.
"""

import pytest
import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut
from options.strategies import (
    bull_call_spread,
    bear_put_spread,
    long_straddle,
    iron_condor
)


class TestOptionsStrategies:
    """Test cases for options strategies."""

    @pytest.fixture
    def setup_options(self):
        """Create sample options for testing."""
        expiration = datetime.date.today() + timedelta(days=30)
        call_100 = EuropeanCall(100, 100, 0.3, expiration, 0.05, 0.1)
        call_105 = EuropeanCall(100, 105, 0.3, expiration, 0.05, 0.1)
        put_95 = EuropeanPut(100, 95, 0.3, expiration, 0.05, 0.1)
        put_100 = EuropeanPut(100, 100, 0.3, expiration, 0.05, 0.1)

        return {
            'call_100': call_100,
            'call_105': call_105,
            'put_95': put_95,
            'put_100': put_100
        }

    def test_bull_call_spread(self, setup_options):
        """Test bull call spread strategy."""
        strategy = bull_call_spread(
            setup_options['call_100'],
            setup_options['call_105']
        )

        assert strategy.name == "Bull Call Spread"
        assert len(strategy.legs) == 2
        # Bull call spread is a debit strategy
        assert strategy.net_cost > 0

    def test_long_straddle(self, setup_options):
        """Test long straddle strategy."""
        strategy = long_straddle(
            setup_options['call_100'],
            setup_options['put_100']
        )

        assert strategy.name == "Long Straddle"
        assert len(strategy.legs) == 2
        # Long straddle costs money (debit)
        assert strategy.net_cost > 0
        # Delta should be near zero for ATM straddle
        assert abs(strategy.net_delta) < 10

    def test_strategy_greeks(self, setup_options):
        """Test that strategy Greeks are calculated."""
        strategy = bull_call_spread(
            setup_options['call_100'],
            setup_options['call_105']
        )

        # Strategy should have Greeks
        assert isinstance(strategy.net_delta, float)
        assert isinstance(strategy.net_gamma, float)
        assert isinstance(strategy.net_vega, float)
        assert isinstance(strategy.net_theta, float)
