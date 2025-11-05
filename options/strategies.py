"""
Options Strategies Module

This module implements common options trading strategies including:
- Vertical spreads (bull/bear call/put spreads)
- Straddles and strangles
- Iron condors and butterflies
- Calendar spreads
- And more...
"""

import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StrategyLeg:
    """
    Represents one leg of an options strategy.

    Attributes:
        option: Option object (EuropeanCall, EuropeanPut, etc.)
        quantity: Number of contracts (positive for long, negative for short)
        action: 'buy' or 'sell'
    """
    option: any
    quantity: float
    action: str  # 'buy' or 'sell'

    @property
    def cost(self) -> float:
        """Cost of this leg (negative for short positions)."""
        multiplier = -1 if self.action == 'sell' else 1
        return self.option.price * abs(self.quantity) * 100 * multiplier

    @property
    def delta(self) -> float:
        """Delta of this leg."""
        multiplier = -1 if self.action == 'sell' else 1
        return self.option.delta * self.quantity * 100 * multiplier


class OptionsStrategy:
    """
    Base class for options strategies.
    """

    def __init__(self, name: str, legs: List[StrategyLeg]):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            legs: List of StrategyLeg objects
        """
        self.name = name
        self.legs = legs

    @property
    def net_cost(self) -> float:
        """Net cost of the strategy (debit if positive, credit if negative)."""
        return sum(leg.cost for leg in self.legs)

    @property
    def net_delta(self) -> float:
        """Net delta of the strategy."""
        return sum(leg.delta for leg in self.legs)

    @property
    def net_gamma(self) -> float:
        """Net gamma of the strategy."""
        multiplier_map = {'buy': 1, 'sell': -1}
        return sum(
            leg.option.gamma * leg.quantity * 100 * multiplier_map[leg.action]
            for leg in self.legs
        )

    @property
    def net_vega(self) -> float:
        """Net vega of the strategy."""
        multiplier_map = {'buy': 1, 'sell': -1}
        return sum(
            leg.option.vega * leg.quantity * 100 * multiplier_map[leg.action]
            for leg in self.legs
        )

    @property
    def net_theta(self) -> float:
        """Net theta of the strategy."""
        multiplier_map = {'buy': 1, 'sell': -1}
        return sum(
            leg.option.theta * leg.quantity * 100 * multiplier_map[leg.action]
            for leg in self.legs
        )

    def payoff_at_expiration(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Calculate strategy payoff at expiration for different stock prices.

        Args:
            stock_prices: Array of stock prices

        Returns:
            Array of payoffs
        """
        payoff = np.zeros_like(stock_prices, dtype=float)

        for leg in self.legs:
            option = leg.option
            quantity = leg.quantity
            multiplier = -1 if leg.action == 'sell' else 1

            # Determine if call or put
            if hasattr(option, '__class__'):
                option_class_name = option.__class__.__name__
                is_call = 'Call' in option_class_name
            else:
                is_call = True  # Default assumption

            # Calculate intrinsic value at each stock price
            if is_call:
                intrinsic = np.maximum(stock_prices - option.strike_price, 0)
            else:
                intrinsic = np.maximum(option.strike_price - stock_prices, 0)

            payoff += intrinsic * quantity * 100 * multiplier

        # Subtract net cost
        payoff -= self.net_cost

        return payoff

    def plot_payoff_diagram(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        num_points: int = 100
    ) -> None:
        """
        Plot payoff diagram for the strategy.

        Args:
            price_range: Tuple of (min_price, max_price) or None for auto
            num_points: Number of points to plot
        """
        # Determine price range if not provided
        if price_range is None:
            strikes = [leg.option.strike_price for leg in self.legs]
            min_strike = min(strikes)
            max_strike = max(strikes)
            price_range = (min_strike * 0.7, max_strike * 1.3)

        # Generate stock prices
        stock_prices = np.linspace(price_range[0], price_range[1], num_points)

        # Calculate payoffs
        payoffs = self.payoff_at_expiration(stock_prices)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, payoffs, 'b-', linewidth=2, label='Payoff at Expiration')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=-self.net_cost, color='r', linestyle='--', alpha=0.3, label='Initial Cost')

        # Mark strikes
        for leg in self.legs:
            plt.axvline(x=leg.option.strike_price, color='g', linestyle=':', alpha=0.5)

        # Mark breakevens
        # Find where payoff crosses zero
        sign_changes = np.diff(np.sign(payoffs))
        breakeven_indices = np.where(sign_changes != 0)[0]
        for idx in breakeven_indices:
            be_price = stock_prices[idx]
            plt.plot(be_price, 0, 'ro', markersize=8)
            plt.annotate(f'BE: ${be_price:.2f}',
                        xy=(be_price, 0),
                        xytext=(be_price, payoffs.max() * 0.1),
                        ha='center',
                        fontsize=9)

        plt.xlabel('Stock Price at Expiration ($)')
        plt.ylabel('Profit/Loss ($)')
        plt.title(f'{self.name} - Payoff Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_summary(self) -> Dict:
        """Get strategy summary."""
        return {
            'name': self.name,
            'num_legs': len(self.legs),
            'net_cost': self.net_cost,
            'net_delta': self.net_delta,
            'net_gamma': self.net_gamma,
            'net_vega': self.net_vega,
            'net_theta': self.net_theta,
            'max_profit': self.calculate_max_profit(),
            'max_loss': self.calculate_max_loss()
        }

    def calculate_max_profit(self) -> Optional[float]:
        """Calculate maximum profit (if bounded)."""
        # This is strategy-specific and would need to be overridden
        return None

    def calculate_max_loss(self) -> Optional[float]:
        """Calculate maximum loss (if bounded)."""
        # This is strategy-specific and would need to be overridden
        return None


# Specific strategy builders

def bull_call_spread(
    long_call: any,
    short_call: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a bull call spread.

    Buy lower strike call, sell higher strike call.
    Bullish strategy with limited profit and limited risk.

    Args:
        long_call: Long call option (lower strike)
        short_call: Short call option (higher strike)
        quantity: Number of spreads

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(long_call, quantity, 'buy'),
        StrategyLeg(short_call, quantity, 'sell')
    ]
    return OptionsStrategy("Bull Call Spread", legs)


def bear_put_spread(
    long_put: any,
    short_put: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a bear put spread.

    Buy higher strike put, sell lower strike put.
    Bearish strategy with limited profit and limited risk.

    Args:
        long_put: Long put option (higher strike)
        short_put: Short put option (lower strike)
        quantity: Number of spreads

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(long_put, quantity, 'buy'),
        StrategyLeg(short_put, quantity, 'sell')
    ]
    return OptionsStrategy("Bear Put Spread", legs)


def long_straddle(
    call: any,
    put: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a long straddle.

    Buy both call and put at same strike.
    Profits from large moves in either direction.

    Args:
        call: Call option
        put: Put option (same strike as call)
        quantity: Number of straddles

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(call, quantity, 'buy'),
        StrategyLeg(put, quantity, 'buy')
    ]
    return OptionsStrategy("Long Straddle", legs)


def short_straddle(
    call: any,
    put: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a short straddle.

    Sell both call and put at same strike.
    Profits from low volatility (stock stays near strike).

    Args:
        call: Call option
        put: Put option (same strike as call)
        quantity: Number of straddles

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(call, quantity, 'sell'),
        StrategyLeg(put, quantity, 'sell')
    ]
    return OptionsStrategy("Short Straddle", legs)


def long_strangle(
    call: any,
    put: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a long strangle.

    Buy call and put at different strikes (call strike > put strike).
    Profits from large moves, cheaper than straddle.

    Args:
        call: Call option (higher strike)
        put: Put option (lower strike)
        quantity: Number of strangles

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(call, quantity, 'buy'),
        StrategyLeg(put, quantity, 'buy')
    ]
    return OptionsStrategy("Long Strangle", legs)


def iron_condor(
    long_put_lower: any,
    short_put: any,
    short_call: any,
    long_call_higher: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create an iron condor.

    Sell put spread and call spread.
    Profits from low volatility in a range.

    Args:
        long_put_lower: Long put (lowest strike)
        short_put: Short put
        short_call: Short call
        long_call_higher: Long call (highest strike)
        quantity: Number of condors

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(long_put_lower, quantity, 'buy'),
        StrategyLeg(short_put, quantity, 'sell'),
        StrategyLeg(short_call, quantity, 'sell'),
        StrategyLeg(long_call_higher, quantity, 'buy')
    ]
    return OptionsStrategy("Iron Condor", legs)


def butterfly_spread(
    long_call_lower: any,
    short_call_middle: any,
    long_call_higher: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a butterfly spread.

    Buy 1 lower strike, sell 2 middle strike, buy 1 higher strike.
    Profits if stock stays near middle strike.

    Args:
        long_call_lower: Long call (lowest strike)
        short_call_middle: Short call (middle strike, 2x quantity)
        long_call_higher: Long call (highest strike)
        quantity: Number of butterflies

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(long_call_lower, quantity, 'buy'),
        StrategyLeg(short_call_middle, 2 * quantity, 'sell'),
        StrategyLeg(long_call_higher, quantity, 'buy')
    ]
    return OptionsStrategy("Butterfly Spread", legs)


def covered_call(
    call: any,
    stock_quantity: int = 100,
    call_quantity: int = 1
) -> OptionsStrategy:
    """
    Create a covered call.

    Own stock and sell call option.
    Generates income while capping upside.

    Args:
        call: Call option to sell
        stock_quantity: Number of shares owned
        call_quantity: Number of calls to sell

    Returns:
        OptionsStrategy object
    """
    # Note: This is simplified. In a full implementation, you'd need a Stock class
    legs = [
        StrategyLeg(call, call_quantity, 'sell')
    ]
    return OptionsStrategy("Covered Call", legs)


def protective_put(
    put: any,
    stock_quantity: int = 100,
    put_quantity: int = 1
) -> OptionsStrategy:
    """
    Create a protective put.

    Own stock and buy put option.
    Insurance against downside risk.

    Args:
        put: Put option to buy
        stock_quantity: Number of shares owned
        put_quantity: Number of puts to buy

    Returns:
        OptionsStrategy object
    """
    # Note: This is simplified. In a full implementation, you'd need a Stock class
    legs = [
        StrategyLeg(put, put_quantity, 'buy')
    ]
    return OptionsStrategy("Protective Put", legs)


def collar(
    put: any,
    call: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a collar.

    Buy put and sell call (when owning stock).
    Limits both upside and downside.

    Args:
        put: Put option to buy (lower strike)
        call: Call option to sell (higher strike)
        quantity: Number of collars

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(put, quantity, 'buy'),
        StrategyLeg(call, quantity, 'sell')
    ]
    return OptionsStrategy("Collar", legs)


def calendar_spread(
    near_option: any,
    far_option: any,
    quantity: int = 1
) -> OptionsStrategy:
    """
    Create a calendar (time) spread.

    Sell near-term option, buy far-term option (same strike).
    Profits from time decay differential.

    Args:
        near_option: Near-term option to sell
        far_option: Far-term option to buy
        quantity: Number of spreads

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(near_option, quantity, 'sell'),
        StrategyLeg(far_option, quantity, 'buy')
    ]
    return OptionsStrategy("Calendar Spread", legs)


def ratio_spread(
    long_option: any,
    short_option: any,
    long_quantity: int = 1,
    short_quantity: int = 2
) -> OptionsStrategy:
    """
    Create a ratio spread.

    Buy options at one strike, sell more options at another strike.

    Args:
        long_option: Option to buy
        short_option: Option to sell
        long_quantity: Number to buy
        short_quantity: Number to sell

    Returns:
        OptionsStrategy object
    """
    legs = [
        StrategyLeg(long_option, long_quantity, 'buy'),
        StrategyLeg(short_option, short_quantity, 'sell')
    ]
    return OptionsStrategy("Ratio Spread", legs)
