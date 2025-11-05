"""
Unit tests for American options pricing.
"""

import pytest
import datetime
from datetime import timedelta
from options.american_options import AmericanCall, AmericanPut, compare_american_european


class TestAmericanCall:
    """Test cases for American call options."""

    @pytest.fixture
    def sample_call(self):
        """Create a sample American call option."""
        expiration = datetime.date.today() + timedelta(days=30)
        return AmericanCall(100.0, 105.0, 0.3, expiration, 0.05, 0.02)

    def test_initialization(self, sample_call):
        """Test that American call initializes correctly."""
        assert sample_call.asset_price == 100.0
        assert sample_call.strike_price == 105.0
        assert sample_call.volatility == 0.3

    def test_price_positive(self, sample_call):
        """Test that option price is positive."""
        assert sample_call.price > 0

    def test_american_vs_european(self):
        """Test that American call >= European call."""
        expiration = datetime.date.today() + timedelta(days=30)
        comparison = compare_american_european(
            100.0, 105.0, 0.3, expiration, 0.05, 0.02, 'call'
        )
        # American should be at least as valuable as European
        assert comparison['american_price'] >= comparison['european_price']


class TestAmericanPut:
    """Test cases for American put options."""

    @pytest.fixture
    def sample_put(self):
        """Create a sample American put option."""
        expiration = datetime.date.today() + timedelta(days=30)
        return AmericanPut(100.0, 105.0, 0.3, expiration, 0.05, 0.02)

    def test_price_positive(self, sample_put):
        """Test that option price is positive."""
        assert sample_put.price > 0

    def test_american_vs_european_put(self):
        """Test that American put >= European put."""
        expiration = datetime.date.today() + timedelta(days=30)
        comparison = compare_american_european(
            100.0, 105.0, 0.3, expiration, 0.05, 0.02, 'put'
        )
        # American put should be more valuable due to early exercise
        assert comparison['american_price'] >= comparison['european_price']
        assert comparison['early_exercise_premium'] >= 0
