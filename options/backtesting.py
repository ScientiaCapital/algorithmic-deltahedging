"""
Backtesting Framework for Options Strategies

This module provides a comprehensive framework for backtesting options
trading strategies using historical data.
"""

import datetime
from datetime import timedelta
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BacktestTrade:
    """
    Record of a single trade in backtest.

    Attributes:
        entry_date: Date trade was entered
        exit_date: Date trade was exited (optional)
        strategy_name: Name of strategy
        entry_price: Entry price
        exit_price: Exit price (optional)
        quantity: Number of contracts/shares
        pnl: Profit/loss (calculated)
        status: 'open' or 'closed'
    """
    entry_date: datetime.date
    strategy_name: str
    entry_price: float
    quantity: float
    exit_date: Optional[datetime.date] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'open'
    metadata: Dict = field(default_factory=dict)

    def close_trade(self, exit_date: datetime.date, exit_price: float) -> float:
        """
        Close the trade and calculate P&L.

        Args:
            exit_date: Exit date
            exit_price: Exit price

        Returns:
            Realized P&L
        """
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.quantity * 100
        self.status = 'closed'
        return self.pnl


class BacktestEngine:
    """
    Main backtesting engine for options strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_contract: float = 0.65,
        slippage_bps: float = 5.0
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission_per_contract: Commission per option contract
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_bps = slippage_bps

        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime.date, float]] = []
        self.performance_metrics: Dict = {}

    def calculate_transaction_costs(
        self,
        price: float,
        quantity: float
    ) -> float:
        """
        Calculate total transaction costs.

        Args:
            price: Option price
            quantity: Number of contracts

        Returns:
            Total transaction costs
        """
        commission = abs(quantity) * self.commission_per_contract
        notional = price * abs(quantity) * 100
        slippage = notional * (self.slippage_bps / 10000)
        return commission + slippage

    def enter_trade(
        self,
        date: datetime.date,
        strategy_name: str,
        price: float,
        quantity: float,
        metadata: Optional[Dict] = None
    ) -> Optional[BacktestTrade]:
        """
        Enter a new trade.

        Args:
            date: Entry date
            strategy_name: Strategy name
            price: Entry price
            quantity: Number of contracts
            metadata: Additional trade metadata

        Returns:
            BacktestTrade object or None if insufficient capital
        """
        # Calculate required capital
        required_capital = price * abs(quantity) * 100
        transaction_costs = self.calculate_transaction_costs(price, quantity)
        total_cost = required_capital + transaction_costs

        # Check if we have enough capital
        if total_cost > self.current_capital:
            return None

        # Create trade
        trade = BacktestTrade(
            entry_date=date,
            strategy_name=strategy_name,
            entry_price=price,
            quantity=quantity,
            metadata=metadata or {}
        )

        # Update capital
        self.current_capital -= total_cost

        # Record trade
        self.trades.append(trade)

        return trade

    def exit_trade(
        self,
        trade: BacktestTrade,
        exit_date: datetime.date,
        exit_price: float
    ) -> float:
        """
        Exit an existing trade.

        Args:
            trade: Trade to exit
            exit_date: Exit date
            exit_price: Exit price

        Returns:
            Realized P&L
        """
        # Close trade
        pnl = trade.close_trade(exit_date, exit_price)

        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(
            exit_price, trade.quantity
        )

        # Update capital
        proceeds = exit_price * abs(trade.quantity) * 100
        self.current_capital += proceeds - transaction_costs

        # Update P&L
        trade.pnl = pnl - transaction_costs

        return trade.pnl

    def record_equity(self, date: datetime.date) -> None:
        """
        Record current equity value.

        Args:
            date: Current date
        """
        # Calculate total equity (capital + value of open positions)
        open_positions_value = sum(
            trade.entry_price * trade.quantity * 100
            for trade in self.trades if trade.status == 'open'
        )

        total_equity = self.current_capital + open_positions_value
        self.equity_curve.append((date, total_equity))

    def run_backtest(
        self,
        strategy_func: Callable,
        historical_data: pd.DataFrame,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        **strategy_params
    ) -> Dict:
        """
        Run backtest with a strategy function.

        Args:
            strategy_func: Function that takes (date, data, engine, **params)
                          and executes trading logic
            historical_data: DataFrame with historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            **strategy_params: Parameters to pass to strategy function

        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            historical_data = historical_data[
                historical_data.index >= pd.Timestamp(start_date)
            ]
        if end_date:
            historical_data = historical_data[
                historical_data.index <= pd.Timestamp(end_date)
            ]

        # Iterate through each date
        for date in historical_data.index:
            # Get data up to current date
            current_data = historical_data.loc[:date]

            # Execute strategy
            strategy_func(date.date(), current_data, self, **strategy_params)

            # Record equity
            self.record_equity(date.date())

        # Calculate performance metrics
        self.calculate_performance_metrics()

        return self.get_results()

    def calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return

        # Extract equity values
        dates, equity_values = zip(*self.equity_curve)
        equity_array = np.array(equity_values)

        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]

        # Total return
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        daily_vol = np.std(returns) if len(returns) > 0 else 0
        annualized_vol = daily_vol * np.sqrt(252)

        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        # Win rate
        closed_trades = [t for t in self.trades if t.status == 'closed']
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

        # Average win/loss
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [t.pnl for t in closed_trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        self.performance_metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_values[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(self.trades) - len(closed_trades),
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commissions': sum(
                self.calculate_transaction_costs(t.entry_price, t.quantity)
                for t in self.trades
            )
        }

    def get_results(self) -> Dict:
        """
        Get backtest results.

        Returns:
            Dictionary with results and metrics
        """
        return {
            'performance_metrics': self.performance_metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

    def plot_results(self) -> None:
        """Plot backtest results."""
        if not self.equity_curve:
            print("No data to plot")
            return

        dates, equity_values = zip(*self.equity_curve)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        ax1.plot(dates, equity_values, 'b-', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5,
                   label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        equity_array = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100

        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self) -> None:
        """Print formatted performance summary."""
        if not self.performance_metrics:
            print("No metrics available")
            return

        metrics = self.performance_metrics

        print("="*70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*70)

        print("\nCAPITAL")
        print("-"*70)
        print(f"Initial Capital:        ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital:          ${metrics['final_capital']:,.2f}")
        print(f"Total Return:           {metrics['total_return']*100:.2f}%")

        print("\nRETURNS & RISK")
        print("-"*70)
        print(f"Annualized Return:      {metrics['annualized_return']*100:.2f}%")
        print(f"Annualized Volatility:  {metrics['annualized_volatility']*100:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown:       {metrics['max_drawdown']*100:.2f}%")

        print("\nTRADING STATISTICS")
        print("-"*70)
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Closed Trades:          {metrics['closed_trades']}")
        print(f"Open Trades:            {metrics['open_trades']}")
        print(f"Win Rate:               {metrics['win_rate']*100:.2f}%")

        print("\nP&L ANALYSIS")
        print("-"*70)
        print(f"Average Win:            ${metrics['average_win']:,.2f}")
        print(f"Average Loss:           ${metrics['average_loss']:,.2f}")
        print(f"Profit Factor:          {metrics['profit_factor']:.2f}")

        print("\nCOSTS")
        print("-"*70)
        print(f"Total Commissions:      ${metrics['total_commissions']:,.2f}")

        print("="*70)


def simple_moving_average_strategy(
    date: datetime.date,
    data: pd.DataFrame,
    engine: BacktestEngine,
    short_window: int = 10,
    long_window: int = 50
) -> None:
    """
    Example strategy: Simple moving average crossover.

    Args:
        date: Current date
        data: Historical data up to current date
        engine: Backtest engine
        short_window: Short MA window
        long_window: Long MA window
    """
    if len(data) < long_window:
        return

    # Calculate moving averages
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

    # Get current values
    current_short_ma = data['SMA_short'].iloc[-1]
    current_long_ma = data['SMA_long'].iloc[-1]
    prev_short_ma = data['SMA_short'].iloc[-2]
    prev_long_ma = data['SMA_long'].iloc[-2]

    current_price = data['Close'].iloc[-1]

    # Find open positions
    open_trades = [t for t in engine.trades if t.status == 'open']

    # Buy signal (short MA crosses above long MA)
    if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
        if not open_trades:  # No open position
            # Enter long position
            engine.enter_trade(
                date=date,
                strategy_name="SMA Crossover",
                price=current_price,
                quantity=1,
                metadata={'signal': 'buy'}
            )

    # Sell signal (short MA crosses below long MA)
    elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
        if open_trades:  # Have open position
            # Exit position
            for trade in open_trades:
                engine.exit_trade(trade, date, current_price)
