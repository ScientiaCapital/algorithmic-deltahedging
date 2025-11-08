"""
Portfolio Management for Options and Hedging Positions

This module provides portfolio-level management, aggregation, and analysis
capabilities for options trading and delta hedging strategies.
"""

import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class PositionType(Enum):
    """Types of positions in the portfolio."""
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"


@dataclass
class Position:
    """
    Represents a single position in the portfolio.

    Attributes:
        symbol: Underlying asset symbol
        position_type: Type of position (call, put, stock, etc.)
        quantity: Number of contracts/shares
        entry_price: Price at which position was entered
        entry_date: Date position was opened
        current_price: Current market price
        option: Reference to option object if applicable
    """
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    entry_date: datetime.date
    current_price: Optional[float] = None
    option: Optional[any] = None  # EuropeanCall or EuropeanPut

    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        price = self.current_price if self.current_price else self.entry_price
        if self.position_type in [PositionType.LONG_CALL, PositionType.SHORT_CALL,
                                  PositionType.LONG_PUT, PositionType.SHORT_PUT]:
            # Options: multiply by 100 (contract multiplier)
            return price * self.quantity * 100
        else:
            # Stock positions
            return price * self.quantity

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis of position."""
        if self.position_type in [PositionType.LONG_CALL, PositionType.SHORT_CALL,
                                  PositionType.LONG_PUT, PositionType.SHORT_PUT]:
            return self.entry_price * self.quantity * 100
        else:
            return self.entry_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.position_type in [PositionType.SHORT_CALL, PositionType.SHORT_PUT,
                                  PositionType.SHORT_STOCK]:
            # Short positions: profit when price decreases
            return self.cost_basis - self.market_value
        else:
            # Long positions: profit when price increases
            return self.market_value - self.cost_basis

    @property
    def delta(self) -> float:
        """Get position delta."""
        if self.option:
            delta_per_contract = self.option.delta * 100
            return delta_per_contract * self.quantity
        elif self.position_type == PositionType.LONG_STOCK:
            return self.quantity
        elif self.position_type == PositionType.SHORT_STOCK:
            return -self.quantity
        else:
            return 0.0

    @property
    def gamma(self) -> float:
        """Get position gamma."""
        if self.option:
            return self.option.gamma * self.quantity * 100
        return 0.0

    @property
    def vega(self) -> float:
        """Get position vega."""
        if self.option:
            return self.option.vega * self.quantity * 100
        return 0.0

    @property
    def theta(self) -> float:
        """Get position theta."""
        if self.option:
            return self.option.theta * self.quantity * 100
        return 0.0


class Portfolio:
    """
    Multi-asset portfolio manager for options and stock positions.

    This class provides:
    - Position tracking and management
    - Portfolio-level Greeks aggregation
    - Risk metrics calculation
    - P&L reporting
    """

    def __init__(self, name: str = "Default Portfolio"):
        """
        Initialize portfolio.

        Args:
            name: Portfolio name/identifier
        """
        self.name = name
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.realized_pnl = 0.0
        self.cash_balance = 0.0
        self.creation_date = datetime.date.today()

    def add_position(self, position: Position) -> None:
        """
        Add a new position to the portfolio.

        Args:
            position: Position object to add

        Raises:
            ValueError: If position is None or if insufficient cash for long positions
        """
        if position is None:
            raise ValueError("Position cannot be None")

        if position.cost_basis is None or position.cost_basis < 0:
            raise ValueError("Position cost_basis must be non-negative")

        # Check if enough cash for long positions
        if position.position_type in [PositionType.LONG_CALL, PositionType.LONG_PUT,
                                     PositionType.LONG_STOCK]:
            required_cash = position.cost_basis
            if self.cash_balance < required_cash:
                raise ValueError(
                    f"Insufficient cash: need ${required_cash:.2f}, "
                    f"have ${self.cash_balance:.2f}"
                )

        self.positions.append(position)

        # Update cash balance (decrease for long, increase for short)
        if position.position_type in [PositionType.LONG_CALL, PositionType.LONG_PUT,
                                     PositionType.LONG_STOCK]:
            self.cash_balance -= position.cost_basis
        else:
            self.cash_balance += position.cost_basis

    def remove_position(self, position: Position, closing_price: float) -> float:
        """
        Close and remove a position from the portfolio.

        Args:
            position: Position to close
            closing_price: Price at which position is closed

        Returns:
            Realized P&L from closing the position

        Raises:
            ValueError: If position not found or closing_price is invalid
        """
        if position is None:
            raise ValueError("Position cannot be None")

        if position not in self.positions:
            raise ValueError("Position not found in portfolio")

        if closing_price is None or closing_price < 0:
            raise ValueError("Closing price must be non-negative")

        # Calculate realized P&L
        position.current_price = closing_price
        realized_pnl = position.unrealized_pnl

        # Update cash balance
        if position.position_type in [PositionType.LONG_CALL, PositionType.LONG_PUT,
                                     PositionType.LONG_STOCK]:
            self.cash_balance += position.market_value
        else:
            self.cash_balance -= position.market_value

        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)
        self.realized_pnl += realized_pnl

        return realized_pnl

    def update_prices(self, price_updates: Dict[str, float]) -> None:
        """
        Update current prices for all positions.

        Args:
            price_updates: Dictionary mapping symbols to current prices
        """
        for position in self.positions:
            if position.symbol in price_updates:
                position.current_price = price_updates[position.symbol]

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a specific symbol.

        Args:
            symbol: Asset symbol

        Returns:
            List of positions for the symbol
        """
        return [pos for pos in self.positions if pos.symbol == symbol]

    def get_positions_by_type(self, position_type: PositionType) -> List[Position]:
        """
        Get all positions of a specific type.

        Args:
            position_type: Type of position to filter

        Returns:
            List of positions matching the type
        """
        return [pos for pos in self.positions if pos.position_type == position_type]

    @property
    def total_market_value(self) -> float:
        """Calculate total market value of all positions."""
        return sum(pos.market_value for pos in self.positions)

    @property
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions)

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.total_unrealized_pnl

    @property
    def portfolio_delta(self) -> float:
        """Calculate portfolio-level delta."""
        return sum(pos.delta for pos in self.positions)

    @property
    def portfolio_gamma(self) -> float:
        """Calculate portfolio-level gamma."""
        return sum(pos.gamma for pos in self.positions)

    @property
    def portfolio_vega(self) -> float:
        """Calculate portfolio-level vega."""
        return sum(pos.vega for pos in self.positions)

    @property
    def portfolio_theta(self) -> float:
        """Calculate portfolio-level theta."""
        return sum(pos.theta for pos in self.positions)

    @property
    def net_asset_value(self) -> float:
        """Calculate net asset value (positions + cash)."""
        return self.total_market_value + self.cash_balance

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR) using delta-normal method.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days

        Returns:
            VaR estimate
        """
        if not self.positions:
            return 0.0

        # Simplified VaR calculation using portfolio delta
        # More sophisticated methods would use full covariance matrix
        portfolio_value = self.total_market_value
        portfolio_delta_pct = self.portfolio_delta / portfolio_value if portfolio_value != 0 else 0

        # Assume average volatility (this should be calculated properly in production)
        avg_volatility = 0.3  # 30% annualized

        # Calculate VaR
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        daily_volatility = avg_volatility / np.sqrt(252)
        time_factor = np.sqrt(time_horizon_days)

        var = abs(portfolio_value * portfolio_delta_pct * daily_volatility * z_score * time_factor)
        return var

    def get_summary(self) -> Dict[str, any]:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio statistics
        """
        return {
            'name': self.name,
            'creation_date': self.creation_date,
            'num_positions': len(self.positions),
            'num_closed_positions': len(self.closed_positions),
            'cash_balance': self.cash_balance,
            'total_market_value': self.total_market_value,
            'net_asset_value': self.net_asset_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.total_unrealized_pnl,
            'total_pnl': self.total_pnl,
            'portfolio_delta': self.portfolio_delta,
            'portfolio_gamma': self.portfolio_gamma,
            'portfolio_vega': self.portfolio_vega,
            'portfolio_theta': self.portfolio_theta,
            'var_95_1day': self.calculate_var(0.95, 1)
        }

    def print_summary(self) -> None:
        """Print formatted portfolio summary."""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"PORTFOLIO SUMMARY: {summary['name']}")
        print(f"{'='*60}")
        print(f"Creation Date:        {summary['creation_date']}")
        print(f"Active Positions:     {summary['num_positions']}")
        print(f"Closed Positions:     {summary['num_closed_positions']}")
        print(f"\nVALUATION")
        print(f"{'-'*60}")
        print(f"Cash Balance:         ${summary['cash_balance']:,.2f}")
        print(f"Market Value:         ${summary['total_market_value']:,.2f}")
        print(f"Net Asset Value:      ${summary['net_asset_value']:,.2f}")
        print(f"\nP&L")
        print(f"{'-'*60}")
        print(f"Realized P&L:         ${summary['realized_pnl']:,.2f}")
        print(f"Unrealized P&L:       ${summary['unrealized_pnl']:,.2f}")
        print(f"Total P&L:            ${summary['total_pnl']:,.2f}")
        print(f"\nGREEKS")
        print(f"{'-'*60}")
        print(f"Portfolio Delta:      {summary['portfolio_delta']:.2f}")
        print(f"Portfolio Gamma:      {summary['portfolio_gamma']:.4f}")
        print(f"Portfolio Vega:       {summary['portfolio_vega']:.2f}")
        print(f"Portfolio Theta:      {summary['portfolio_theta']:.2f}")
        print(f"\nRISK METRICS")
        print(f"{'-'*60}")
        print(f"1-Day VaR (95%):      ${summary['var_95_1day']:,.2f}")
        print(f"{'='*60}\n")

    def get_positions_table(self) -> List[Dict[str, any]]:
        """
        Get detailed table of all positions.

        Returns:
            List of dictionaries with position details
        """
        positions_data = []
        for pos in self.positions:
            positions_data.append({
                'symbol': pos.symbol,
                'type': pos.position_type.value,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'delta': pos.delta,
                'gamma': pos.gamma,
                'vega': pos.vega,
                'theta': pos.theta
            })
        return positions_data
