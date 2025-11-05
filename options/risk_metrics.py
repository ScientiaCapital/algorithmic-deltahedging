"""
Risk Metrics Module

Advanced risk metric calculations including:
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Conditional Value at Risk (CVaR) / Expected Shortfall
- Maximum Drawdown
- Sharpe Ratio
- Other portfolio risk measures
"""

import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class RiskMetrics:
    """
    Calculate various risk metrics for portfolios and positions.
    """

    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk using historical simulation method.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days

        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        # Scale returns for time horizon if needed
        if time_horizon_days != 1:
            returns = returns * np.sqrt(time_horizon_days)

        # Calculate percentile
        alpha = 1 - confidence_level
        var = -np.percentile(returns, alpha * 100)

        return float(var)

    @staticmethod
    def parametric_var(
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk using parametric (variance-covariance) method.

        Assumes returns are normally distributed.

        Args:
            portfolio_value: Current portfolio value
            expected_return: Expected daily return (as decimal)
            volatility: Daily volatility (as decimal)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days

        Returns:
            VaR value (positive number representing potential loss)
        """
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # Scale for time horizon
        time_factor = np.sqrt(time_horizon_days)

        # VaR calculation
        var = portfolio_value * (
            -expected_return * time_horizon_days +
            volatility * time_factor * z_score
        )

        return float(var)

    @staticmethod
    def monte_carlo_var(
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate Value at Risk using Monte Carlo simulation.

        Args:
            portfolio_value: Current portfolio value
            expected_return: Expected daily return (as decimal)
            volatility: Daily volatility (as decimal)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (VaR value, array of simulated returns)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Simulate portfolio returns
        simulated_returns = np.random.normal(
            expected_return * time_horizon_days,
            volatility * np.sqrt(time_horizon_days),
            num_simulations
        )

        # Calculate portfolio values
        simulated_portfolio_values = portfolio_value * (1 + simulated_returns)

        # Calculate losses
        losses = portfolio_value - simulated_portfolio_values

        # VaR is the (1 - confidence_level) percentile of losses
        var = np.percentile(losses, (1 - confidence_level) * 100)

        return float(var), simulated_returns

    @staticmethod
    def conditional_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall.

        CVaR is the expected loss given that the loss exceeds VaR.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days

        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0

        # Scale returns for time horizon if needed
        if time_horizon_days != 1:
            returns = returns * np.sqrt(time_horizon_days)

        # Calculate VaR threshold
        alpha = 1 - confidence_level
        var_threshold = -np.percentile(returns, alpha * 100)

        # CVaR is the mean of all losses exceeding VaR
        losses = -returns
        tail_losses = losses[losses >= var_threshold]

        if len(tail_losses) == 0:
            return var_threshold

        cvar = np.mean(tail_losses)
        return float(cvar)

    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown from peak.

        Args:
            prices: Array of historical prices or portfolio values

        Returns:
            Dictionary with max drawdown metrics
        """
        if len(prices) == 0:
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0, 'peak': 0.0, 'trough': 0.0}

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = abs(drawdown[max_dd_idx])

        # Find peak before max drawdown
        peak_idx = np.argmax(prices[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        peak_value = prices[peak_idx]
        trough_value = prices[max_dd_idx]

        return {
            'max_drawdown': float(peak_value - trough_value),
            'max_drawdown_pct': float(max_dd),
            'peak': float(peak_value),
            'trough': float(trough_value),
            'peak_date_idx': int(peak_idx),
            'trough_date_idx': int(max_dd_idx)
        }

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns - rf_period

        # Sharpe ratio
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)

        return float(sharpe)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only considers downside volatility).

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns - rf_period

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        # Sortino ratio
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)

        return float(sortino)

    @staticmethod
    def beta(
        asset_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> float:
        """
        Calculate beta (systematic risk relative to market).

        Args:
            asset_returns: Array of asset returns
            market_returns: Array of market returns

        Returns:
            Beta coefficient
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) == 0:
            return 0.0

        # Covariance / Variance
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance
        return float(beta)


class PortfolioRiskAnalyzer:
    """
    Comprehensive risk analysis for portfolios.
    """

    def __init__(
        self,
        portfolio_values: np.ndarray,
        returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize risk analyzer.

        Args:
            portfolio_values: Array of historical portfolio values
            returns: Array of returns (calculated if not provided)
            risk_free_rate: Annual risk-free rate
        """
        self.portfolio_values = portfolio_values
        self.risk_free_rate = risk_free_rate

        if returns is not None:
            self.returns = returns
        else:
            # Calculate returns from values
            self.returns = np.diff(portfolio_values) / portfolio_values[:-1]

    def calculate_all_metrics(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizon: int = 1
    ) -> Dict:
        """
        Calculate all risk metrics.

        Args:
            confidence_levels: List of confidence levels for VaR
            time_horizon: Time horizon in days

        Returns:
            Dictionary with all risk metrics
        """
        current_value = self.portfolio_values[-1] if len(self.portfolio_values) > 0 else 0
        mean_return = np.mean(self.returns) if len(self.returns) > 0 else 0
        volatility = np.std(self.returns) if len(self.returns) > 0 else 0

        metrics = {
            'current_value': float(current_value),
            'mean_daily_return': float(mean_return),
            'daily_volatility': float(volatility),
            'annualized_return': float(mean_return * 252),
            'annualized_volatility': float(volatility * np.sqrt(252))
        }

        # VaR at different confidence levels
        for conf in confidence_levels:
            conf_pct = int(conf * 100)

            # Historical VaR
            hist_var = RiskMetrics.historical_var(self.returns, conf, time_horizon)
            metrics[f'historical_var_{conf_pct}'] = hist_var

            # Parametric VaR
            param_var = RiskMetrics.parametric_var(
                current_value, mean_return, volatility, conf, time_horizon
            )
            metrics[f'parametric_var_{conf_pct}'] = param_var

            # Monte Carlo VaR
            mc_var, _ = RiskMetrics.monte_carlo_var(
                current_value, mean_return, volatility, conf, time_horizon
            )
            metrics[f'monte_carlo_var_{conf_pct}'] = mc_var

            # CVaR
            cvar = RiskMetrics.conditional_var(self.returns, conf, time_horizon)
            metrics[f'cvar_{conf_pct}'] = cvar * current_value

        # Maximum Drawdown
        mdd = RiskMetrics.maximum_drawdown(self.portfolio_values)
        metrics['max_drawdown'] = mdd['max_drawdown']
        metrics['max_drawdown_pct'] = mdd['max_drawdown_pct']

        # Sharpe Ratio
        metrics['sharpe_ratio'] = RiskMetrics.sharpe_ratio(self.returns, self.risk_free_rate)

        # Sortino Ratio
        metrics['sortino_ratio'] = RiskMetrics.sortino_ratio(self.returns, self.risk_free_rate)

        return metrics

    def generate_risk_report(self) -> str:
        """
        Generate formatted risk report.

        Returns:
            Formatted string with risk metrics
        """
        metrics = self.calculate_all_metrics()

        report = []
        report.append("="*70)
        report.append("PORTFOLIO RISK ANALYSIS REPORT")
        report.append("="*70)

        report.append("\nPORTFOLIO STATISTICS")
        report.append("-"*70)
        report.append(f"Current Value:          ${metrics['current_value']:,.2f}")
        report.append(f"Mean Daily Return:      {metrics['mean_daily_return']*100:.4f}%")
        report.append(f"Daily Volatility:       {metrics['daily_volatility']*100:.4f}%")
        report.append(f"Annualized Return:      {metrics['annualized_return']*100:.2f}%")
        report.append(f"Annualized Volatility:  {metrics['annualized_volatility']*100:.2f}%")

        report.append("\nVALUE AT RISK (1-DAY)")
        report.append("-"*70)
        report.append(f"Historical VaR (95%):   ${metrics.get('historical_var_95', 0):,.2f}")
        report.append(f"Parametric VaR (95%):   ${metrics.get('parametric_var_95', 0):,.2f}")
        report.append(f"Monte Carlo VaR (95%):  ${metrics.get('monte_carlo_var_95', 0):,.2f}")
        report.append(f"Historical VaR (99%):   ${metrics.get('historical_var_99', 0):,.2f}")
        report.append(f"Parametric VaR (99%):   ${metrics.get('parametric_var_99', 0):,.2f}")
        report.append(f"Monte Carlo VaR (99%):  ${metrics.get('monte_carlo_var_99', 0):,.2f}")

        report.append("\nCONDITIONAL VALUE AT RISK (CVaR/ES)")
        report.append("-"*70)
        report.append(f"CVaR (95%):             ${metrics.get('cvar_95', 0):,.2f}")
        report.append(f"CVaR (99%):             ${metrics.get('cvar_99', 0):,.2f}")

        report.append("\nDRAWDOWN ANALYSIS")
        report.append("-"*70)
        report.append(f"Maximum Drawdown:       ${metrics['max_drawdown']:,.2f}")
        report.append(f"Maximum Drawdown %:     {metrics['max_drawdown_pct']*100:.2f}%")

        report.append("\nRISK-ADJUSTED RETURNS")
        report.append("-"*70)
        report.append(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
        report.append(f"Sortino Ratio:          {metrics['sortino_ratio']:.4f}")

        report.append("="*70)

        return "\n".join(report)


def stress_test_portfolio(
    portfolio_value: float,
    positions_deltas: List[float],
    price_shocks: List[float],
    volatility_shocks: List[float] = [0.0]
) -> pd.DataFrame:
    """
    Perform stress testing on portfolio.

    Args:
        portfolio_value: Current portfolio value
        positions_deltas: List of delta values for positions
        price_shocks: List of price shocks to test (e.g., [-0.10, -0.05, 0, 0.05, 0.10])
        volatility_shocks: List of volatility shocks to test

    Returns:
        DataFrame with stress test results
    """
    results = []

    total_delta = sum(positions_deltas)

    for price_shock in price_shocks:
        for vol_shock in volatility_shocks:
            # Simplified P&L calculation
            # In reality, you would reprice all options with new parameters
            delta_pnl = total_delta * price_shock * portfolio_value / 100

            new_value = portfolio_value + delta_pnl
            pnl_pct = (new_value - portfolio_value) / portfolio_value * 100

            results.append({
                'price_shock': price_shock,
                'volatility_shock': vol_shock,
                'portfolio_value': new_value,
                'pnl': delta_pnl,
                'pnl_pct': pnl_pct
            })

    return pd.DataFrame(results)
