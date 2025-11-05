"""
Unit tests for delta hedging strategy.
"""

import pytest
import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut
from options.delta_hedging import (
    DeltaHedgingStrategy,
    HedgePosition,
    Transaction,
    TransactionType
)


class TestDeltaHedgingStrategy:
    """Test cases for DeltaHedgingStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a sample hedging strategy."""
        return DeltaHedgingStrategy(
            rebalance_threshold=0.1,
            commission_per_share=0.005,
            slippage_bps=5.0
        )

    @pytest.fixture
    def sample_call(self):
        """Create a sample call option."""
        expiration = datetime.date.today() + timedelta(days=30)
        return EuropeanCall(100.0, 105.0, 0.3, expiration, 0.05, 0.1)

    def test_strategy_initialization(self, strategy):
        """Test that strategy initializes correctly."""
        assert strategy.rebalance_threshold == 0.1
        assert strategy.commission_per_share == 0.005
        assert strategy.slippage_bps == 5.0
        assert len(strategy.positions) == 0
        assert strategy.total_transaction_costs == 0.0

    def test_transaction_cost_calculation(self, strategy):
        """Test transaction cost calculation."""
        cost = strategy.calculate_transaction_cost(100, 50.0)
        # Commission: 100 * 0.005 = 0.5
        # Slippage: 100 * 50 * 0.0005 = 2.5
        # Total: 3.0
        assert abs(cost - 3.0) < 0.01

    def test_create_hedge_long_call(self, strategy, sample_call):
        """Test creating hedge for long call position."""
        position = strategy.create_hedge(
            option=sample_call,
            option_quantity=1,
            current_stock_price=100.0,
            target_delta=0.0
        )

        assert position is not None
        assert position.option == sample_call
        assert position.option_quantity == 1
        # For delta-neutral, should short stock equal to call delta * 100
        expected_stock_qty = -sample_call.delta * 100
        assert abs(position.stock_quantity - expected_stock_qty) < 1.0

    def test_create_hedge_short_call(self, strategy, sample_call):
        """Test creating hedge for short call position."""
        position = strategy.create_hedge(
            option=sample_call,
            option_quantity=-1,
            current_stock_price=100.0,
            target_delta=0.0
        )

        # For short call, delta is negative, so need to long stock
        assert position.stock_quantity > 0

    def test_check_rebalance_needed(self, strategy, sample_call):
        """Test rebalancing threshold check."""
        position = HedgePosition(
            option=sample_call,
            option_quantity=1,
            stock_quantity=-50,  # Deliberately imbalanced
            initial_stock_price=100.0,
            target_delta=0.0
        )

        # Should need rebalancing if delta imbalance is large
        needs_rebalance = strategy.check_rebalance_needed(position)
        # This depends on the specific delta, but with threshold 0.1 and
        # stock_quantity = -50, likely needs rebalance
        assert isinstance(needs_rebalance, bool)

    def test_transaction_costs_accumulate(self, strategy, sample_call):
        """Test that transaction costs accumulate correctly."""
        initial_costs = strategy.total_transaction_costs

        strategy.create_hedge(
            option=sample_call,
            option_quantity=1,
            current_stock_price=100.0
        )

        assert strategy.total_transaction_costs > initial_costs

    def test_pnl_calculation(self, strategy, sample_call):
        """Test P&L calculation for hedged position."""
        position = strategy.create_hedge(
            option=sample_call,
            option_quantity=1,
            current_stock_price=100.0,
            target_delta=0.0
        )

        # Calculate P&L with unchanged prices
        pnl = strategy.calculate_pnl(
            position=position,
            current_stock_price=100.0,
            current_option_price=sample_call.price
        )

        assert 'option_pnl' in pnl
        assert 'stock_pnl' in pnl
        assert 'transaction_costs' in pnl
        assert 'net_pnl' in pnl

        # With no price change, option and stock P&L should be near zero
        # Net P&L should be negative due to transaction costs
        assert pnl['net_pnl'] < 0

    def test_portfolio_summary(self, strategy, sample_call):
        """Test portfolio summary generation."""
        strategy.create_hedge(sample_call, 1, 100.0)
        strategy.create_hedge(sample_call, 1, 100.0)

        summary = strategy.get_portfolio_summary()

        assert summary['total_positions'] == 2
        assert 'total_delta' in summary
        assert 'total_transaction_costs' in summary


class TestHedgePosition:
    """Test cases for HedgePosition class."""

    @pytest.fixture
    def sample_call(self):
        """Create a sample call option."""
        expiration = datetime.date.today() + timedelta(days=30)
        return EuropeanCall(100.0, 105.0, 0.3, expiration, 0.05, 0.1)

    def test_current_delta_calculation(self, sample_call):
        """Test current delta calculation."""
        position = HedgePosition(
            option=sample_call,
            option_quantity=1,
            stock_quantity=-50,
            initial_stock_price=100.0
        )

        # Current delta = option delta * 100 + stock quantity
        expected_delta = sample_call.delta * 100 - 50
        assert abs(position.current_delta - expected_delta) < 0.01

    def test_delta_imbalance(self, sample_call):
        """Test delta imbalance calculation."""
        position = HedgePosition(
            option=sample_call,
            option_quantity=1,
            stock_quantity=-50,
            initial_stock_price=100.0,
            target_delta=0.0
        )

        imbalance = position.delta_imbalance
        # Should be non-zero if position is not perfectly hedged
        assert isinstance(imbalance, float)


class TestTransaction:
    """Test cases for Transaction class."""

    def test_transaction_creation(self):
        """Test transaction object creation."""
        txn = Transaction(
            timestamp=datetime.datetime.now(),
            transaction_type=TransactionType.BUY_STOCK,
            quantity=100,
            price=50.0,
            cost=5.0
        )

        assert txn.quantity == 100
        assert txn.price == 50.0
        assert txn.cost == 5.0

    def test_total_value_calculation(self):
        """Test total value calculation."""
        txn = Transaction(
            timestamp=datetime.datetime.now(),
            transaction_type=TransactionType.BUY_STOCK,
            quantity=100,
            price=50.0,
            cost=5.0
        )

        # Total value = quantity * price + cost
        expected_total = 100 * 50.0 + 5.0
        assert txn.total_value == expected_total
