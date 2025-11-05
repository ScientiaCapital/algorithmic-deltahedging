"""
Unit tests for implied volatility calculations.
"""

import pytest
import math
from options.implied_volatility import (
    implied_volatility,
    implied_volatility_newton_raphson,
    black_scholes_price,
    ImpliedVolatilityError
)


class TestImpliedVolatility:
    """Test cases for implied volatility calculation."""

    def test_round_trip(self):
        """Test that IV calculation is consistent with BS pricing."""
        # Price an option with known volatility
        S, K, r, T = 100.0, 105.0, 0.05, 0.25
        true_vol = 0.30

        # Calculate theoretical price
        price = black_scholes_price(S, K, true_vol, T, r, 'call')

        # Calculate implied volatility
        calculated_iv = implied_volatility(
            price, S, K, T, r, 'call', method='newton'
        )

        # Should match original volatility
        assert abs(calculated_iv - true_vol) < 0.01

    def test_different_methods_agree(self):
        """Test that different IV methods give similar results."""
        S, K, r, T = 100.0, 100.0, 0.05, 1.0
        market_price = 10.0

        iv_newton = implied_volatility(market_price, S, K, T, r, 'call', method='newton')
        iv_brent = implied_volatility(market_price, S, K, T, r, 'call', method='brent')

        # Different methods should agree within tolerance
        assert abs(iv_newton - iv_brent) < 0.02

    def test_invalid_price_raises_error(self):
        """Test that invalid option price raises error."""
        with pytest.raises(ImpliedVolatilityError):
            # Price below intrinsic value
            implied_volatility(-1.0, 100, 105, 0.25, 0.05, 'call')

    def test_put_option_iv(self):
        """Test implied volatility for put options."""
        S, K, r, T = 100.0, 105.0, 0.05, 0.25
        true_vol = 0.25

        price = black_scholes_price(S, K, true_vol, T, r, 'put')
        calculated_iv = implied_volatility(price, S, K, T, r, 'put')

        assert abs(calculated_iv - true_vol) < 0.01
