"""
Delta Hedging Strategy Implementation

This module implements delta-neutral hedging strategies for option portfolios,
including rebalancing logic, transaction cost modeling, and P&L tracking.
"""

import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class TransactionType(Enum):
    """Types of transactions in the hedging portfolio."""
    BUY_STOCK = "buy_stock"
    SELL_STOCK = "sell_stock"
    BUY_OPTION = "buy_option"
    SELL_OPTION = "sell_option"


@dataclass
class Transaction:
    """
    Record of a single transaction in the portfolio.

    Attributes:
        timestamp: When the transaction occurred
        transaction_type: Type of transaction
        quantity: Number of units traded
        price: Price per unit
        cost: Total transaction cost (commissions + slippage)
    """
    timestamp: datetime.datetime
    transaction_type: TransactionType
    quantity: float
    price: float
    cost: float = 0.0

    @property
    def total_value(self) -> float:
        """Total value of the transaction including costs."""
        return self.quantity * self.price + self.cost


@dataclass
class HedgePosition:
    """
    Current hedging position for an option.

    Attributes:
        option: The option being hedged
        option_quantity: Number of option contracts held (positive for long)
        stock_quantity: Number of shares held for hedging
        initial_stock_price: Stock price when position was initiated
        target_delta: Target portfolio delta (usually 0 for delta-neutral)
    """
    option: any  # EuropeanCall or EuropeanPut
    option_quantity: float
    stock_quantity: float
    initial_stock_price: float
    target_delta: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)

    @property
    def current_delta(self) -> float:
        """Calculate current portfolio delta."""
        option_delta = self.option.delta * self.option_quantity * 100  # 100 shares per contract
        stock_delta = self.stock_quantity
        return option_delta + stock_delta

    @property
    def delta_imbalance(self) -> float:
        """Calculate how far the current delta is from target."""
        return self.current_delta - self.target_delta


class DeltaHedgingStrategy:
    """
    Delta-neutral hedging strategy implementation.

    This class manages delta hedging of option positions, including:
    - Initial hedge setup
    - Rebalancing triggers and execution
    - Transaction cost tracking
    - P&L calculation
    """

    def __init__(
        self,
        rebalance_threshold: float = 0.1,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0,
        min_rebalance_shares: int = 1
    ):
        """
        Initialize delta hedging strategy.

        Args:
            rebalance_threshold: Trigger rebalancing when |delta| exceeds this (e.g., 0.1 = 10 shares)
            commission_per_share: Commission cost per share traded
            slippage_bps: Slippage in basis points (100 bps = 1%)
            min_rebalance_shares: Minimum number of shares to trigger rebalancing
        """
        self.rebalance_threshold = rebalance_threshold
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        self.min_rebalance_shares = min_rebalance_shares
        self.positions: List[HedgePosition] = []
        self.total_transaction_costs = 0.0

    def calculate_transaction_cost(self, quantity: float, price: float) -> float:
        """
        Calculate total transaction cost including commission and slippage.

        Args:
            quantity: Number of shares to trade (absolute value)
            price: Price per share

        Returns:
            Total transaction cost
        """
        abs_quantity = abs(quantity)
        commission = abs_quantity * self.commission_per_share
        slippage = abs_quantity * price * (self.slippage_bps / 10000)
        return commission + slippage

    def create_hedge(
        self,
        option: any,
        option_quantity: float,
        current_stock_price: float,
        target_delta: float = 0.0
    ) -> HedgePosition:
        """
        Create initial delta hedge for an option position.

        Args:
            option: EuropeanCall or EuropeanPut object
            option_quantity: Number of option contracts (positive for long, negative for short)
            current_stock_price: Current price of the underlying stock
            target_delta: Target portfolio delta (default 0 for delta-neutral)

        Returns:
            HedgePosition object representing the hedge
        """
        # Calculate required stock position for delta neutrality
        # Each option contract typically represents 100 shares
        option_delta_exposure = option.delta * option_quantity * 100

        # To neutralize: stock_quantity + option_delta_exposure = target_delta
        required_stock_quantity = target_delta - option_delta_exposure

        # Create transaction record
        transaction_cost = self.calculate_transaction_cost(
            abs(required_stock_quantity), current_stock_price
        )

        transaction_type = (
            TransactionType.BUY_STOCK if required_stock_quantity > 0
            else TransactionType.SELL_STOCK
        )

        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            transaction_type=transaction_type,
            quantity=abs(required_stock_quantity),
            price=current_stock_price,
            cost=transaction_cost
        )

        self.total_transaction_costs += transaction_cost

        # Create hedge position
        position = HedgePosition(
            option=option,
            option_quantity=option_quantity,
            stock_quantity=required_stock_quantity,
            initial_stock_price=current_stock_price,
            target_delta=target_delta,
            transactions=[transaction]
        )

        self.positions.append(position)
        return position

    def check_rebalance_needed(self, position: HedgePosition) -> bool:
        """
        Check if position needs rebalancing based on delta imbalance.

        Args:
            position: HedgePosition to check

        Returns:
            True if rebalancing is needed
        """
        delta_imbalance = abs(position.delta_imbalance)
        return bool(delta_imbalance >= self.rebalance_threshold)

    def rebalance_position(
        self,
        position: HedgePosition,
        current_stock_price: float
    ) -> Optional[Transaction]:
        """
        Rebalance a position to restore delta neutrality.

        Args:
            position: HedgePosition to rebalance
            current_stock_price: Current stock price

        Returns:
            Transaction record if rebalancing occurred, None otherwise
        """
        if not self.check_rebalance_needed(position):
            return None

        # Calculate required adjustment
        current_option_delta = position.option.delta * position.option_quantity * 100
        required_stock_quantity = position.target_delta - current_option_delta
        adjustment_needed = required_stock_quantity - position.stock_quantity

        # Check if adjustment meets minimum threshold
        if abs(adjustment_needed) < self.min_rebalance_shares:
            return None

        # Create transaction
        transaction_cost = self.calculate_transaction_cost(
            abs(adjustment_needed), current_stock_price
        )

        transaction_type = (
            TransactionType.BUY_STOCK if adjustment_needed > 0
            else TransactionType.SELL_STOCK
        )

        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            transaction_type=transaction_type,
            quantity=abs(adjustment_needed),
            price=current_stock_price,
            cost=transaction_cost
        )

        # Update position
        position.stock_quantity += adjustment_needed
        position.transactions.append(transaction)
        self.total_transaction_costs += transaction_cost

        return transaction

    def calculate_pnl(
        self,
        position: HedgePosition,
        current_stock_price: float,
        current_option_price: float
    ) -> Dict[str, float]:
        """
        Calculate profit and loss for a hedged position.

        Args:
            position: HedgePosition to analyze
            current_stock_price: Current stock price
            current_option_price: Current option price

        Returns:
            Dictionary with PnL breakdown
        """
        # Option P&L
        initial_option_value = position.option.price * position.option_quantity * 100
        current_option_value = current_option_price * position.option_quantity * 100
        option_pnl = current_option_value - initial_option_value

        # Stock P&L
        # Calculate average cost of stock position from transactions
        stock_cost = 0.0
        for txn in position.transactions:
            if txn.transaction_type in [TransactionType.BUY_STOCK, TransactionType.SELL_STOCK]:
                multiplier = 1 if txn.transaction_type == TransactionType.BUY_STOCK else -1
                stock_cost += multiplier * txn.quantity * txn.price

        current_stock_value = position.stock_quantity * current_stock_price
        stock_pnl = current_stock_value - stock_cost

        # Transaction costs
        total_costs = sum(txn.cost for txn in position.transactions)

        # Net P&L
        net_pnl = option_pnl + stock_pnl - total_costs

        return {
            'option_pnl': option_pnl,
            'stock_pnl': stock_pnl,
            'transaction_costs': total_costs,
            'net_pnl': net_pnl,
            'current_delta': position.current_delta
        }

    def get_portfolio_summary(self) -> Dict[str, any]:
        """
        Get summary statistics for all positions.

        Returns:
            Dictionary with portfolio-level statistics
        """
        total_positions = len(self.positions)
        total_delta = sum(pos.current_delta for pos in self.positions)

        return {
            'total_positions': total_positions,
            'total_delta': total_delta,
            'total_transaction_costs': self.total_transaction_costs,
            'positions': self.positions
        }


class DynamicHedgingSimulator:
    """
    Simulator for testing delta hedging strategies over time.

    This class simulates asset price movements and triggers rebalancing
    to evaluate hedging strategy performance.
    """

    def __init__(
        self,
        strategy: DeltaHedgingStrategy,
        initial_stock_price: float,
        volatility: float,
        drift: float,
        dt: float = 1/252
    ):
        """
        Initialize hedging simulator.

        Args:
            strategy: DeltaHedgingStrategy to simulate
            initial_stock_price: Starting stock price
            volatility: Annualized volatility
            drift: Expected return (drift)
            dt: Time step in years (default: 1 trading day)
        """
        self.strategy = strategy
        self.initial_stock_price = initial_stock_price
        self.current_stock_price = initial_stock_price
        self.volatility = volatility
        self.drift = drift
        self.dt = dt
        self.price_history: List[Tuple[datetime.datetime, float]] = []
        self.pnl_history: List[Dict[str, float]] = []

    def simulate_price_step(self) -> float:
        """
        Simulate one time step of stock price movement using GBM.

        Returns:
            New stock price
        """
        z = np.random.normal(0, 1)
        price_change = self.current_stock_price * np.exp(
            (self.drift - 0.5 * self.volatility**2) * self.dt +
            self.volatility * np.sqrt(self.dt) * z
        )
        self.current_stock_price = price_change
        self.price_history.append((datetime.datetime.now(), self.current_stock_price))
        return self.current_stock_price

    def run_simulation(
        self,
        position: HedgePosition,
        num_steps: int,
        record_interval: int = 1
    ) -> List[Dict[str, any]]:
        """
        Run hedging simulation for specified number of time steps.

        Args:
            position: HedgePosition to simulate
            num_steps: Number of time steps to simulate
            record_interval: Record P&L every N steps

        Returns:
            List of simulation results at each recording interval
        """
        results = []

        for step in range(num_steps):
            # Simulate price movement
            new_stock_price = self.simulate_price_step()

            # Update option object with new price (simplified - would need full recalculation)
            # This is a placeholder - in reality you'd recreate the option with new parameters

            # Check if rebalancing is needed
            transaction = self.strategy.rebalance_position(position, new_stock_price)

            # Record results at specified intervals
            if step % record_interval == 0:
                pnl = self.strategy.calculate_pnl(
                    position,
                    new_stock_price,
                    position.option.price  # Would need to recalculate in real scenario
                )
                results.append({
                    'step': step,
                    'stock_price': new_stock_price,
                    'rebalanced': transaction is not None,
                    **pnl
                })

        return results
