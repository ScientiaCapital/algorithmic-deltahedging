"""
Exotic Options Pricing Module

This module implements pricing for exotic options including:
- Asian options (average price/strike)
- Barrier options (knock-in/knock-out)
- Lookback options
- Digital (binary) options
"""

import math
from typing import Optional, List
import numpy as np
from scipy import stats


class AsianOption:
    """
    Asian Option (Average Price Option)

    The payoff depends on the average price of the underlying over
    a specified period. Priced using Monte Carlo simulation.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        time_to_expiration: float,
        risk_free_rate: float,
        option_type: str = 'call',
        averaging_type: str = 'arithmetic',
        num_simulations: int = 10000,
        num_steps: int = 252
    ):
        """
        Initialize Asian option.

        Args:
            asset_price: Current asset price
            strike_price: Strike price
            volatility: Volatility
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            averaging_type: 'arithmetic' or 'geometric'
            num_simulations: Number of Monte Carlo simulations
            num_steps: Number of time steps for averaging
        """
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type.lower()
        self.averaging_type = averaging_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        # Price the option
        self.price, self.std_error = self._monte_carlo_price()

    def _monte_carlo_price(self) -> tuple:
        """
        Price Asian option using Monte Carlo simulation.

        Returns:
            Tuple of (price, standard_error)
        """
        dt = self.time_to_expiration / self.num_steps
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiration)

        payoffs = []

        for _ in range(self.num_simulations):
            # Simulate price path
            prices = [self.asset_price]

            for step in range(self.num_steps):
                z = np.random.normal(0, 1)
                drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
                diffusion = self.volatility * math.sqrt(dt) * z
                next_price = prices[-1] * math.exp(drift + diffusion)
                prices.append(next_price)

            # Calculate average
            if self.averaging_type == 'arithmetic':
                avg_price = np.mean(prices)
            else:  # geometric
                avg_price = np.exp(np.mean(np.log(prices)))

            # Calculate payoff
            if self.option_type == 'call':
                payoff = max(avg_price - self.strike_price, 0)
            else:  # put
                payoff = max(self.strike_price - avg_price, 0)

            payoffs.append(payoff)

        # Calculate price and standard error
        payoffs = np.array(payoffs)
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / math.sqrt(self.num_simulations)

        return price, std_error


class BarrierOption:
    """
    Barrier Option (Knock-In/Knock-Out)

    The option activates (knock-in) or deactivates (knock-out) when
    the underlying price crosses a barrier level.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        barrier_level: float,
        volatility: float,
        time_to_expiration: float,
        risk_free_rate: float,
        option_type: str = 'call',
        barrier_type: str = 'down-and-out',
        num_simulations: int = 10000,
        num_steps: int = 252
    ):
        """
        Initialize barrier option.

        Args:
            asset_price: Current asset price
            strike_price: Strike price
            barrier_level: Barrier price level
            volatility: Volatility
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            barrier_type: 'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
            num_simulations: Number of Monte Carlo simulations
            num_steps: Number of time steps for monitoring
        """
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.barrier_level = barrier_level
        self.volatility = volatility
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        # Price the option
        self.price, self.std_error = self._monte_carlo_price()

    def _check_barrier(self, prices: List[float]) -> bool:
        """
        Check if barrier was crossed.

        Args:
            prices: List of prices along the path

        Returns:
            True if barrier condition is met
        """
        if 'down' in self.barrier_type:
            crossed = any(p <= self.barrier_level for p in prices)
        else:  # up
            crossed = any(p >= self.barrier_level for p in prices)

        if 'out' in self.barrier_type:
            # Knock-out: option becomes worthless if barrier crossed
            return not crossed
        else:  # in
            # Knock-in: option only activates if barrier crossed
            return crossed

    def _monte_carlo_price(self) -> tuple:
        """
        Price barrier option using Monte Carlo simulation.

        Returns:
            Tuple of (price, standard_error)
        """
        dt = self.time_to_expiration / self.num_steps
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiration)

        payoffs = []

        for _ in range(self.num_simulations):
            # Simulate price path
            prices = [self.asset_price]

            for step in range(self.num_steps):
                z = np.random.normal(0, 1)
                drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
                diffusion = self.volatility * math.sqrt(dt) * z
                next_price = prices[-1] * math.exp(drift + diffusion)
                prices.append(next_price)

            # Check if barrier condition is met
            if self._check_barrier(prices):
                # Calculate standard option payoff
                final_price = prices[-1]
                if self.option_type == 'call':
                    payoff = max(final_price - self.strike_price, 0)
                else:  # put
                    payoff = max(self.strike_price - final_price, 0)
            else:
                payoff = 0

            payoffs.append(payoff)

        # Calculate price and standard error
        payoffs = np.array(payoffs)
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / math.sqrt(self.num_simulations)

        return price, std_error


class LookbackOption:
    """
    Lookback Option

    The payoff depends on the maximum or minimum price reached
    during the option's life.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: Optional[float],
        volatility: float,
        time_to_expiration: float,
        risk_free_rate: float,
        option_type: str = 'call',
        lookback_type: str = 'floating',
        num_simulations: int = 10000,
        num_steps: int = 252
    ):
        """
        Initialize lookback option.

        Args:
            asset_price: Current asset price
            strike_price: Strike price (None for floating strike)
            volatility: Volatility
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            lookback_type: 'floating' or 'fixed'
            num_simulations: Number of Monte Carlo simulations
            num_steps: Number of time steps
        """
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type.lower()
        self.lookback_type = lookback_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        # Price the option
        self.price, self.std_error = self._monte_carlo_price()

    def _monte_carlo_price(self) -> tuple:
        """
        Price lookback option using Monte Carlo simulation.

        Returns:
            Tuple of (price, standard_error)
        """
        dt = self.time_to_expiration / self.num_steps
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiration)

        payoffs = []

        for _ in range(self.num_simulations):
            # Simulate price path
            prices = [self.asset_price]

            for step in range(self.num_steps):
                z = np.random.normal(0, 1)
                drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
                diffusion = self.volatility * math.sqrt(dt) * z
                next_price = prices[-1] * math.exp(drift + diffusion)
                prices.append(next_price)

            final_price = prices[-1]
            max_price = max(prices)
            min_price = min(prices)

            # Calculate payoff based on type
            if self.lookback_type == 'floating':
                if self.option_type == 'call':
                    # Payoff = S_T - S_min
                    payoff = final_price - min_price
                else:  # put
                    # Payoff = S_max - S_T
                    payoff = max_price - final_price
            else:  # fixed strike
                if self.option_type == 'call':
                    # Payoff = max(S_max - K, 0)
                    payoff = max(max_price - self.strike_price, 0)
                else:  # put
                    # Payoff = max(K - S_min, 0)
                    payoff = max(self.strike_price - min_price, 0)

            payoffs.append(payoff)

        # Calculate price and standard error
        payoffs = np.array(payoffs)
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / math.sqrt(self.num_simulations)

        return price, std_error


class DigitalOption:
    """
    Digital (Binary) Option

    Pays a fixed amount if the option expires in the money, otherwise nothing.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        payout: float,
        volatility: float,
        time_to_expiration: float,
        risk_free_rate: float,
        option_type: str = 'call'
    ):
        """
        Initialize digital option.

        Args:
            asset_price: Current asset price
            strike_price: Strike price
            payout: Fixed payout amount if in the money
            volatility: Volatility
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
        """
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.payout = payout
        self.volatility = volatility
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type.lower()

        # Price using closed-form solution
        self.price = self._analytical_price()

    def _analytical_price(self) -> float:
        """
        Price digital option using closed-form solution.

        Returns:
            Option price
        """
        if self.time_to_expiration <= 0:
            if self.option_type == 'call':
                return self.payout if self.asset_price > self.strike_price else 0
            else:
                return self.payout if self.asset_price < self.strike_price else 0

        # Calculate d2
        d2 = (math.log(self.asset_price / self.strike_price) +
              (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_to_expiration) / \
             (self.volatility * math.sqrt(self.time_to_expiration))

        # Discount factor
        discount = math.exp(-self.risk_free_rate * self.time_to_expiration)

        # Calculate price
        if self.option_type == 'call':
            # Probability of S_T > K
            prob = stats.norm.cdf(d2)
        else:  # put
            # Probability of S_T < K
            prob = stats.norm.cdf(-d2)

        price = self.payout * discount * prob
        return price
