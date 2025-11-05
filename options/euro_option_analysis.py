"""
European Options Pricing and Analysis Module

This module implements Black-Scholes pricing for European call and put options,
along with real-time visualization capabilities for option Greeks and price movements.
"""

import math
import datetime
from datetime import timedelta
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats


class EuropeanCall:
    """
    European Call Option pricing using Black-Scholes model.

    Attributes:
        asset_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the option
        volatility (float): Annualized volatility of the underlying asset
        expiration_date (datetime.date): Expiration date of the option
        risk_free_rate (float): Risk-free interest rate (annualized)
        drift (float): Expected drift rate of the underlying asset
        dt (float): Time to expiration in years
        price (float): Calculated option price
        delta (float): Option delta (sensitivity to underlying price)
    """

    def d1(self, asset_price: float, strike_price: float, risk_free_rate: float,
           volatility: float, dt: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        if dt <= 0:
            raise ValueError("Time to expiration must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        return (math.log(asset_price / strike_price) +
                (risk_free_rate + volatility**2 / 2) * dt) / (volatility * math.sqrt(dt))

    def d2(self, d1: float, volatility: float, dt: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return d1 - volatility * math.sqrt(dt)

    def price(self, asset_price: float, d1: float, strike_price: float,
              d2: float, risk_free_rate: float, dt: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.

        Returns:
            float: The theoretical price of the call option
        """
        n1 = stats.norm.cdf(d1)
        n2 = stats.norm.cdf(d2)
        return asset_price * n1 - strike_price * math.exp(-risk_free_rate * dt) * n2

    def delta(self, d1: float) -> float:
        """
        Calculate option delta (sensitivity to underlying price changes).

        Returns:
            float: Delta value between 0 and 1 for call options
        """
        return stats.norm.cdf(d1)

    def gamma(self, asset_price: float, volatility: float, dt: float, d1: float) -> float:
        """
        Calculate option gamma (rate of change of delta with respect to price).

        Returns:
            float: Gamma value (always positive for both calls and puts)
        """
        if dt <= 0 or volatility <= 0 or asset_price <= 0:
            return 0.0
        return stats.norm.pdf(d1) / (asset_price * volatility * math.sqrt(dt))

    def vega(self, asset_price: float, dt: float, d1: float) -> float:
        """
        Calculate option vega (sensitivity to volatility changes).

        Returns:
            float: Vega value (sensitivity per 1% change in volatility)
        """
        if dt <= 0:
            return 0.0
        return asset_price * stats.norm.pdf(d1) * math.sqrt(dt) / 100

    def theta(self, asset_price: float, strike_price: float, risk_free_rate: float,
              volatility: float, dt: float, d1: float, d2: float) -> float:
        """
        Calculate option theta (time decay - value lost per day).

        Returns:
            float: Theta value (typically negative for long options)
        """
        if dt <= 0:
            return 0.0

        term1 = -(asset_price * stats.norm.pdf(d1) * volatility) / (2 * math.sqrt(dt))
        term2 = risk_free_rate * strike_price * math.exp(-risk_free_rate * dt) * stats.norm.cdf(d2)
        return (term1 - term2) / 365  # Per day

    def rho(self, strike_price: float, risk_free_rate: float, dt: float, d2: float) -> float:
        """
        Calculate option rho (sensitivity to interest rate changes).

        Returns:
            float: Rho value (sensitivity per 1% change in interest rate)
        """
        if dt <= 0:
            return 0.0
        return strike_price * dt * math.exp(-risk_free_rate * dt) * stats.norm.cdf(d2) / 100

    def exercise_prob(self) -> float:
        """
        Calculate probability of option being exercised at expiration.

        Returns:
            float: Probability between 0 and 1
        """
        if self.dt <= 0:
            return 1.0 if self.asset_price > self.strike_price else 0.0

        numerator = (self.strike_price - self.asset_price) - (self.drift * self.asset_price * self.dt)
        denominator = (self.volatility * self.asset_price) * (self.dt ** 0.5)
        return 1 - stats.norm.cdf(numerator / denominator)

    def __init__(self, asset_price: float, strike_price: float, volatility: float,
                 expiration_date: datetime.date, risk_free_rate: float, drift: float):
        """
        Initialize a European Call Option.

        Args:
            asset_price: Current price of the underlying asset (must be positive)
            strike_price: Strike price of the option (must be positive)
            volatility: Annualized volatility (must be positive)
            expiration_date: Expiration date of the option
            risk_free_rate: Annual risk-free interest rate
            drift: Expected drift rate of the underlying asset

        Raises:
            ValueError: If any price/volatility parameters are non-positive
        """
        if asset_price <= 0:
            raise ValueError("Asset price must be positive")
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.drift = drift

        # Calculate time to expiration in years (using business days)
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        self.dt = dt

        if dt > 0:
            # Calculate d1 and d2
            d1 = self.d1(asset_price, strike_price, risk_free_rate, volatility, dt)
            d2 = self.d2(d1, volatility, dt)

            # Calculate option price and all Greeks
            self.price = self.price(asset_price, d1, strike_price, d2, risk_free_rate, dt)
            self.delta = self.delta(d1)
            self.gamma = self.gamma(asset_price, volatility, dt, d1)
            self.vega = self.vega(asset_price, dt, d1)
            self.theta = self.theta(asset_price, strike_price, risk_free_rate,
                                   volatility, dt, d1, d2)
            self.rho = self.rho(strike_price, risk_free_rate, dt, d2)
        else:
            # Option has expired
            self.price = max(0, asset_price - strike_price)
            self.delta = 1.0 if asset_price > strike_price else 0.0
            self.gamma = 0.0
            self.vega = 0.0
            self.theta = 0.0
            self.rho = 0.0


class EuropeanPut:
    """
    European Put Option pricing using Black-Scholes model.

    Attributes:
        asset_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the option
        volatility (float): Annualized volatility of the underlying asset
        expiration_date (datetime.date): Expiration date of the option
        risk_free_rate (float): Risk-free interest rate (annualized)
        drift (float): Expected drift rate of the underlying asset
        dt (float): Time to expiration in years
        price (float): Calculated option price
        delta (float): Option delta (sensitivity to underlying price)
    """

    def d1(self, asset_price: float, strike_price: float, risk_free_rate: float,
           volatility: float, dt: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        if dt <= 0:
            raise ValueError("Time to expiration must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        return (math.log(asset_price / strike_price) +
                (risk_free_rate + volatility**2 / 2) * dt) / (volatility * math.sqrt(dt))

    def d2(self, d1: float, volatility: float, dt: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return d1 - volatility * math.sqrt(dt)

    def price(self, asset_price: float, d1: float, strike_price: float,
              d2: float, risk_free_rate: float, dt: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.

        Returns:
            float: The theoretical price of the put option
        """
        n1 = stats.norm.cdf(-d1)
        n2 = stats.norm.cdf(-d2)
        return strike_price * math.exp(-risk_free_rate * dt) * n2 - asset_price * n1

    def delta(self, d1: float) -> float:
        """
        Calculate option delta (sensitivity to underlying price changes).

        Returns:
            float: Delta value between -1 and 0 for put options
        """
        return stats.norm.cdf(d1) - 1

    def gamma(self, asset_price: float, volatility: float, dt: float, d1: float) -> float:
        """
        Calculate option gamma (rate of change of delta with respect to price).

        Returns:
            float: Gamma value (always positive for both calls and puts)
        """
        if dt <= 0 or volatility <= 0 or asset_price <= 0:
            return 0.0
        return stats.norm.pdf(d1) / (asset_price * volatility * math.sqrt(dt))

    def vega(self, asset_price: float, dt: float, d1: float) -> float:
        """
        Calculate option vega (sensitivity to volatility changes).

        Returns:
            float: Vega value (sensitivity per 1% change in volatility)
        """
        if dt <= 0:
            return 0.0
        return asset_price * stats.norm.pdf(d1) * math.sqrt(dt) / 100

    def theta(self, asset_price: float, strike_price: float, risk_free_rate: float,
              volatility: float, dt: float, d1: float, d2: float) -> float:
        """
        Calculate option theta (time decay - value lost per day).

        Returns:
            float: Theta value (typically negative for long options)
        """
        if dt <= 0:
            return 0.0

        term1 = -(asset_price * stats.norm.pdf(d1) * volatility) / (2 * math.sqrt(dt))
        term2 = risk_free_rate * strike_price * math.exp(-risk_free_rate * dt) * stats.norm.cdf(-d2)
        return (term1 + term2) / 365  # Per day

    def rho(self, strike_price: float, risk_free_rate: float, dt: float, d2: float) -> float:
        """
        Calculate option rho (sensitivity to interest rate changes).

        Returns:
            float: Rho value (sensitivity per 1% change in interest rate)
        """
        if dt <= 0:
            return 0.0
        return -strike_price * dt * math.exp(-risk_free_rate * dt) * stats.norm.cdf(-d2) / 100

    def exercise_prob(self) -> float:
        """
        Calculate probability of option being exercised at expiration.

        Returns:
            float: Probability between 0 and 1
        """
        if self.dt <= 0:
            return 1.0 if self.asset_price < self.strike_price else 0.0

        numerator = (self.strike_price - self.asset_price) - (self.drift * self.asset_price * self.dt)
        denominator = (self.volatility * self.asset_price) * (self.dt ** 0.5)
        return stats.norm.cdf(numerator / denominator)

    def __init__(self, asset_price: float, strike_price: float, volatility: float,
                 expiration_date: datetime.date, risk_free_rate: float, drift: float):
        """
        Initialize a European Put Option.

        Args:
            asset_price: Current price of the underlying asset (must be positive)
            strike_price: Strike price of the option (must be positive)
            volatility: Annualized volatility (must be positive)
            expiration_date: Expiration date of the option
            risk_free_rate: Annual risk-free interest rate
            drift: Expected drift rate of the underlying asset

        Raises:
            ValueError: If any price/volatility parameters are non-positive
        """
        if asset_price <= 0:
            raise ValueError("Asset price must be positive")
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.drift = drift

        # Calculate time to expiration in years (using business days)
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        self.dt = dt

        if dt > 0:
            # Calculate d1 and d2
            d1 = self.d1(asset_price, strike_price, risk_free_rate, volatility, dt)
            d2 = self.d2(d1, volatility, dt)

            # Calculate option price and all Greeks
            self.price = self.price(asset_price, d1, strike_price, d2, risk_free_rate, dt)
            self.delta = self.delta(d1)
            self.gamma = self.gamma(asset_price, volatility, dt, d1)
            self.vega = self.vega(asset_price, dt, d1)
            self.theta = self.theta(asset_price, strike_price, risk_free_rate,
                                   volatility, dt, d1, d2)
            self.rho = self.rho(strike_price, risk_free_rate, dt, d2)
        else:
            # Option has expired
            self.price = max(0, strike_price - asset_price)
            self.delta = -1.0 if asset_price < strike_price else 0.0
            self.gamma = 0.0
            self.vega = 0.0
            self.theta = 0.0
            self.rho = 0.0


class LiveOptionsGraph:
    """
    Real-time visualization of European option pricing and Greeks.

    This class creates an animated plot showing option price, delta, and
    underlying asset price movements over time using simulated GBM price paths.
    """

    def time_step(self, z: int) -> None:
        """
        Update the plot with a new time step.

        This method simulates asset price movement using Geometric Brownian Motion
        and updates all three subplots with new option metrics.

        Args:
            z: Frame number (used by matplotlib FuncAnimation)
        """
        # Calculate time to expiration
        dt = np.busday_count(datetime.date.today(), self.expiration_date) / 252

        if dt > 0:
            # Simulate asset price using Geometric Brownian Motion
            # S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt*Z)
            current_price = self.asset_prices[self.index]
            dt_step = 1/252  # One trading day
            z_rand = np.random.normal(0, 1)
            new_price = current_price * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt_step +
                self.volatility * np.sqrt(dt_step) * z_rand
            )

            # Create new option with updated price
            if self.type == 'call':
                eo = EuropeanCall(new_price, self.strike_price, self.volatility,
                                self.expiration_date, self.risk_free_rate, self.drift)
            elif self.type == 'put':
                eo = EuropeanPut(new_price, self.strike_price, self.volatility,
                               self.expiration_date, self.risk_free_rate, self.drift)
            else:
                return
            # Store new values
            self.option_prices.append(eo.price)
            self.deltas.append(eo.delta)
            self.asset_prices.append(eo.asset_price)
            self.index_set.append(self.index)

            # Clear and update plots
            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            # Plot option price
            self.axs[0].plot(self.index_set, self.option_prices,
                           label='Black-Scholes Option Price', c='b')
            self.axs[0].set_ylabel('Option Price ($)')

            # Plot delta
            self.axs[1].plot(self.index_set, self.deltas, label='Delta', c='gray')
            self.axs[1].set_ylabel('Delta')

            # Plot asset price with ITM/OTM color coding
            current_price = self.asset_prices[self.index]
            if self.type == 'call':
                # Green if ITM (price > strike), red if OTM
                color = 'g' if current_price >= self.strike_price else 'r'
                self.axs[2].plot(self.index_set, self.asset_prices,
                               label='Asset Price', c=color)
                self.axs[2].axhline(y=self.strike_price,
                                  label='Call Strike Price', c='gray', linestyle='--')
            elif self.type == 'put':
                # Green if ITM (price < strike), red if OTM
                color = 'g' if current_price <= self.strike_price else 'r'
                self.axs[2].plot(self.index_set, self.asset_prices,
                               label='Asset Price', c=color)
                self.axs[2].axhline(y=self.strike_price,
                                  label='Put Strike Price', c='gray', linestyle='--')

            self.axs[2].set_ylabel('Asset Price ($)')
            self.axs[2].set_xlabel('Time Steps')

            # Add legends
            self.axs[0].legend(loc='upper left')
            self.axs[1].legend(loc='upper left')
            self.axs[2].legend(loc='upper left')

            # Increment counter and simulate time decay
            self.index += 1
            self.expiration_date = self.expiration_date - timedelta(days=1)

    def __init__(self, european_option, option_type: str):
        """
        Initialize the live options graph visualization.

        Args:
            european_option: An instance of EuropeanCall or EuropeanPut
            option_type: Either 'call' or 'put'
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")

        self.index = 0
        self.asset_price = european_option.asset_price
        self.strike_price = european_option.strike_price
        self.volatility = european_option.volatility
        self.expiration_date = european_option.expiration_date
        self.risk_free_rate = european_option.risk_free_rate
        self.drift = european_option.drift
        self.type = option_type

        # Initialize data storage
        self.index_set = []
        self.option_prices = []
        self.asset_prices = [european_option.asset_price]
        self.deltas = []

        # Set up the plot
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(3, figsize=(10, 8))
        self.fig.suptitle(f'European {option_type.capitalize()} Option Analysis')

        # Create animation
        self.ani = FuncAnimation(plt.gcf(), self.time_step, frames=100,
                                interval=100, repeat=False)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage: Create and visualize a European call option
    # Note: Update the expiration date to a future date for proper functionality
    import sys

    # Calculate a date 30 days in the future
    future_date = datetime.date.today() + timedelta(days=30)

    print("Creating European Call Option...")
    print(f"Asset Price: $64.50")
    print(f"Strike Price: $65.00")
    print(f"Volatility: 40%")
    print(f"Expiration: {future_date}")
    print(f"Risk-free Rate: 6%")
    print(f"Drift: 20%")

    initial_ec = EuropeanCall(64.5, 65, 0.4, future_date, 0.06, 0.2)
    print(f"\n{'='*50}")
    print("OPTION GREEKS AND METRICS")
    print(f"{'='*50}")
    print(f"Option Price: ${initial_ec.price:.2f}")
    print(f"Delta:        {initial_ec.delta:.4f}")
    print(f"Gamma:        {initial_ec.gamma:.4f}")
    print(f"Vega:         {initial_ec.vega:.4f}")
    print(f"Theta:        {initial_ec.theta:.4f} (per day)")
    print(f"Rho:          {initial_ec.rho:.4f}")
    print(f"Exercise Probability: {initial_ec.exercise_prob():.2%}")
    print(f"{'='*50}")

    # Uncomment the line below to see the live visualization
    # lg = LiveOptionsGraph(initial_ec, 'call')
