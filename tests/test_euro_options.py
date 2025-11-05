"""
Unit tests for European options pricing and Greeks calculations.
"""

import pytest
import datetime
from datetime import timedelta
import math
from options.euro_option_analysis import EuropeanCall, EuropeanPut


class TestEuropeanCall:
    """Test cases for EuropeanCall class."""

    @pytest.fixture
    def sample_call(self):
        """Create a sample call option for testing."""
        expiration = datetime.date.today() + timedelta(days=30)
        return EuropeanCall(
            asset_price=100.0,
            strike_price=105.0,
            volatility=0.3,
            expiration_date=expiration,
            risk_free_rate=0.05,
            drift=0.1
        )

    def test_initialization(self, sample_call):
        """Test that call option initializes correctly."""
        assert sample_call.asset_price == 100.0
        assert sample_call.strike_price == 105.0
        assert sample_call.volatility == 0.3
        assert sample_call.risk_free_rate == 0.05
        assert sample_call.drift == 0.1

    def test_price_positive(self, sample_call):
        """Test that option price is positive."""
        assert sample_call.price > 0

    def test_delta_range(self, sample_call):
        """Test that delta is between 0 and 1 for calls."""
        assert 0 <= sample_call.delta <= 1

    def test_gamma_positive(self, sample_call):
        """Test that gamma is positive."""
        assert sample_call.gamma >= 0

    def test_vega_positive(self, sample_call):
        """Test that vega is positive."""
        assert sample_call.vega >= 0

    def test_theta_negative(self, sample_call):
        """Test that theta is typically negative for long calls."""
        # Theta is usually negative for long options (time decay)
        assert sample_call.theta < 0

    def test_rho_positive(self, sample_call):
        """Test that rho is positive for calls."""
        # Call options benefit from higher interest rates
        assert sample_call.rho > 0

    def test_invalid_asset_price(self):
        """Test that negative asset price raises error."""
        expiration = datetime.date.today() + timedelta(days=30)
        with pytest.raises(ValueError):
            EuropeanCall(-100, 105, 0.3, expiration, 0.05, 0.1)

    def test_invalid_strike_price(self):
        """Test that negative strike price raises error."""
        expiration = datetime.date.today() + timedelta(days=30)
        with pytest.raises(ValueError):
            EuropeanCall(100, -105, 0.3, expiration, 0.05, 0.1)

    def test_invalid_volatility(self):
        """Test that negative volatility raises error."""
        expiration = datetime.date.today() + timedelta(days=30)
        with pytest.raises(ValueError):
            EuropeanCall(100, 105, -0.3, expiration, 0.05, 0.1)

    def test_itm_call_higher_price(self):
        """Test that ITM call is more expensive than OTM call."""
        expiration = datetime.date.today() + timedelta(days=30)
        itm_call = EuropeanCall(110, 100, 0.3, expiration, 0.05, 0.1)
        otm_call = EuropeanCall(90, 100, 0.3, expiration, 0.05, 0.1)
        assert itm_call.price > otm_call.price

    def test_higher_volatility_higher_price(self):
        """Test that higher volatility increases option price."""
        expiration = datetime.date.today() + timedelta(days=30)
        low_vol = EuropeanCall(100, 105, 0.2, expiration, 0.05, 0.1)
        high_vol = EuropeanCall(100, 105, 0.4, expiration, 0.05, 0.1)
        assert high_vol.price > low_vol.price


class TestEuropeanPut:
    """Test cases for EuropeanPut class."""

    @pytest.fixture
    def sample_put(self):
        """Create a sample put option for testing."""
        expiration = datetime.date.today() + timedelta(days=30)
        return EuropeanPut(
            asset_price=100.0,
            strike_price=95.0,
            volatility=0.3,
            expiration_date=expiration,
            risk_free_rate=0.05,
            drift=0.1
        )

    def test_initialization(self, sample_put):
        """Test that put option initializes correctly."""
        assert sample_put.asset_price == 100.0
        assert sample_put.strike_price == 95.0
        assert sample_put.volatility == 0.3

    def test_price_positive(self, sample_put):
        """Test that option price is positive."""
        assert sample_put.price > 0

    def test_delta_range(self, sample_put):
        """Test that delta is between -1 and 0 for puts."""
        assert -1 <= sample_put.delta <= 0

    def test_gamma_positive(self, sample_put):
        """Test that gamma is positive."""
        assert sample_put.gamma >= 0

    def test_vega_positive(self, sample_put):
        """Test that vega is positive."""
        assert sample_put.vega >= 0

    def test_rho_negative(self, sample_put):
        """Test that rho is negative for puts."""
        # Put options lose value with higher interest rates
        assert sample_put.rho < 0

    def test_itm_put_higher_price(self):
        """Test that ITM put is more expensive than OTM put."""
        expiration = datetime.date.today() + timedelta(days=30)
        itm_put = EuropeanPut(90, 100, 0.3, expiration, 0.05, 0.1)
        otm_put = EuropeanPut(110, 100, 0.3, expiration, 0.05, 0.1)
        assert itm_put.price > otm_put.price

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        expiration = datetime.date.today() + timedelta(days=30)
        S = 100.0
        K = 100.0
        vol = 0.3
        r = 0.05
        drift = 0.1

        call = EuropeanCall(S, K, vol, expiration, r, drift)
        put = EuropeanPut(S, K, vol, expiration, r, drift)

        # Put-Call Parity: C - P = S - K*e^(-rT)
        import numpy as np
        dt = np.busday_count(datetime.date.today(), expiration) / 252
        parity_diff = call.price - put.price
        theoretical_diff = S - K * math.exp(-r * dt)

        # Allow for small numerical differences
        assert abs(parity_diff - theoretical_diff) < 0.1


class TestExpiredOptions:
    """Test cases for expired options."""

    def test_expired_call_otm(self):
        """Test expired OTM call has zero value."""
        expiration = datetime.date.today() - timedelta(days=1)
        call = EuropeanCall(95, 100, 0.3, expiration, 0.05, 0.1)
        assert call.price == 0

    def test_expired_call_itm(self):
        """Test expired ITM call has intrinsic value."""
        expiration = datetime.date.today() - timedelta(days=1)
        call = EuropeanCall(105, 100, 0.3, expiration, 0.05, 0.1)
        assert call.price == 5

    def test_expired_put_otm(self):
        """Test expired OTM put has zero value."""
        expiration = datetime.date.today() - timedelta(days=1)
        put = EuropeanPut(105, 100, 0.3, expiration, 0.05, 0.1)
        assert put.price == 0

    def test_expired_put_itm(self):
        """Test expired ITM put has intrinsic value."""
        expiration = datetime.date.today() - timedelta(days=1)
        put = EuropeanPut(95, 100, 0.3, expiration, 0.05, 0.1)
        assert put.price == 5
