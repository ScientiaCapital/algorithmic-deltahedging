"""
American Options Pricing Module

This module implements American option pricing using the binomial tree method.
American options can be exercised at any time before expiration, requiring
numerical methods for valuation.
"""

import math
import datetime
from typing import Optional, Tuple
import numpy as np
from scipy import stats


class AmericanOption:
    """
    Base class for American options using binomial tree pricing.

    The binomial tree method discretizes time and builds a tree of possible
    asset prices, then works backward from expiration to determine option value.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        expiration_date: datetime.date,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        num_steps: int = 100,
        calculate_greeks: bool = True
    ):
        """
        Initialize an American option.

        Args:
            asset_price: Current price of the underlying asset
            strike_price: Strike price of the option
            volatility: Annualized volatility
            expiration_date: Expiration date
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual dividend yield
            num_steps: Number of time steps in binomial tree
            calculate_greeks: Whether to calculate Greeks (set False for performance)

        Raises:
            ValueError: If any parameters are invalid
        """
        if asset_price <= 0:
            raise ValueError("Asset price must be positive")
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1")

        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.num_steps = num_steps
        self.calculate_greeks = calculate_greeks

        # Calculate time to expiration in years
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        self.dt = max(dt, 0)

    def _build_binomial_tree(self, option_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build binomial tree for option pricing.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Tuple of (asset_tree, option_tree)
        """
        if self.dt <= 0:
            # Option expired
            if option_type == 'call':
                return np.array([[self.asset_price]]), np.array([[max(0, self.asset_price - self.strike_price)]])
            else:
                return np.array([[self.asset_price]]), np.array([[max(0, self.strike_price - self.asset_price)]])

        # Time step
        dt_step = self.dt / self.num_steps

        # Calculate up and down factors
        u = math.exp(self.volatility * math.sqrt(dt_step))
        d = 1 / u

        # Risk-neutral probability
        a = math.exp((self.risk_free_rate - self.dividend_yield) * dt_step)
        p = (a - d) / (u - d)

        # Build asset price tree
        asset_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                asset_tree[j, i] = self.asset_price * (u ** (i - j)) * (d ** j)

        # Build option value tree (work backward from expiration)
        option_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))

        # Terminal payoffs
        for j in range(self.num_steps + 1):
            if option_type == 'call':
                option_tree[j, self.num_steps] = max(0, asset_tree[j, self.num_steps] - self.strike_price)
            else:  # put
                option_tree[j, self.num_steps] = max(0, self.strike_price - asset_tree[j, self.num_steps])

        # Backward induction
        discount = math.exp(-self.risk_free_rate * dt_step)
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                # Expected value if held
                hold_value = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

                # Exercise value
                if option_type == 'call':
                    exercise_value = max(0, asset_tree[j, i] - self.strike_price)
                else:  # put
                    exercise_value = max(0, self.strike_price - asset_tree[j, i])

                # American option: take maximum of hold vs exercise
                option_tree[j, i] = max(hold_value, exercise_value)

        return asset_tree, option_tree

    def _calculate_greeks(self, option_type: str) -> dict:
        """
        Calculate option Greeks using finite differences on the binomial tree.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary of Greek values
        """
        if self.dt <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }

        # Delta: (V(S+dS) - V(S-dS)) / (2*dS)
        dS = self.asset_price * 0.01

        # Price with S + dS (skip Greeks calculation to avoid infinite recursion)
        option_up = AmericanCall(self.asset_price + dS, self.strike_price, self.volatility,
                                 self.expiration_date, self.risk_free_rate, self.dividend_yield,
                                 self.num_steps, calculate_greeks=False) if option_type == 'call' else \
                    AmericanPut(self.asset_price + dS, self.strike_price, self.volatility,
                               self.expiration_date, self.risk_free_rate, self.dividend_yield,
                               self.num_steps, calculate_greeks=False)

        # Price with S - dS (skip Greeks calculation to avoid infinite recursion)
        option_down = AmericanCall(self.asset_price - dS, self.strike_price, self.volatility,
                                   self.expiration_date, self.risk_free_rate, self.dividend_yield,
                                   self.num_steps, calculate_greeks=False) if option_type == 'call' else \
                      AmericanPut(self.asset_price - dS, self.strike_price, self.volatility,
                                 self.expiration_date, self.risk_free_rate, self.dividend_yield,
                                 self.num_steps, calculate_greeks=False)

        delta = (option_up.price - option_down.price) / (2 * dS)

        # Gamma: (V(S+dS) - 2V(S) + V(S-dS)) / (dS^2)
        gamma = (option_up.price - 2 * self.price + option_down.price) / (dS ** 2)

        # Vega: dV/d(sigma) (skip Greeks calculation to avoid infinite recursion)
        dvol = self.volatility * 0.01
        option_vega = AmericanCall(self.asset_price, self.strike_price, self.volatility + dvol,
                                   self.expiration_date, self.risk_free_rate, self.dividend_yield,
                                   self.num_steps, calculate_greeks=False) if option_type == 'call' else \
                      AmericanPut(self.asset_price, self.strike_price, self.volatility + dvol,
                                 self.expiration_date, self.risk_free_rate, self.dividend_yield,
                                 self.num_steps, calculate_greeks=False)
        vega = (option_vega.price - self.price) / dvol / 100  # Per 1% change

        # Theta: approximate using time step (skip Greeks calculation to avoid infinite recursion)
        if self.dt > 1/252:
            future_date = self.expiration_date - datetime.timedelta(days=1)
            option_theta = AmericanCall(self.asset_price, self.strike_price, self.volatility,
                                       future_date, self.risk_free_rate, self.dividend_yield,
                                       self.num_steps, calculate_greeks=False) if option_type == 'call' else \
                          AmericanPut(self.asset_price, self.strike_price, self.volatility,
                                     future_date, self.risk_free_rate, self.dividend_yield,
                                     self.num_steps, calculate_greeks=False)
            theta = -(option_theta.price - self.price)  # Per day
        else:
            theta = 0.0

        # Rho: dV/dr (skip Greeks calculation to avoid infinite recursion)
        dr = 0.01
        option_rho = AmericanCall(self.asset_price, self.strike_price, self.volatility,
                                 self.expiration_date, self.risk_free_rate + dr, self.dividend_yield,
                                 self.num_steps, calculate_greeks=False) if option_type == 'call' else \
                     AmericanPut(self.asset_price, self.strike_price, self.volatility,
                                self.expiration_date, self.risk_free_rate + dr, self.dividend_yield,
                                self.num_steps, calculate_greeks=False)
        rho = (option_rho.price - self.price) / dr / 100  # Per 1% change

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }


class AmericanCall(AmericanOption):
    """American Call Option using binomial tree pricing."""

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        expiration_date: datetime.date,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        num_steps: int = 100,
        calculate_greeks: bool = True
    ):
        """Initialize American Call option."""
        super().__init__(asset_price, strike_price, volatility, expiration_date,
                        risk_free_rate, dividend_yield, num_steps, calculate_greeks)

        # Calculate price using binomial tree
        _, option_tree = self._build_binomial_tree('call')
        self.price = option_tree[0, 0]

        # Calculate Greeks only if requested
        if calculate_greeks:
            greeks = self._calculate_greeks('call')
            self.delta = greeks['delta']
            self.gamma = greeks['gamma']
            self.vega = greeks['vega']
            self.theta = greeks['theta']
            self.rho = greeks['rho']
        else:
            # Set Greeks to None when not calculated
            self.delta = None
            self.gamma = None
            self.vega = None
            self.theta = None
            self.rho = None


class AmericanPut(AmericanOption):
    """American Put Option using binomial tree pricing."""

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        expiration_date: datetime.date,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        num_steps: int = 100,
        calculate_greeks: bool = True
    ):
        """Initialize American Put option."""
        super().__init__(asset_price, strike_price, volatility, expiration_date,
                        risk_free_rate, dividend_yield, num_steps, calculate_greeks)

        # Calculate price using binomial tree
        _, option_tree = self._build_binomial_tree('put')
        self.price = option_tree[0, 0]

        # Calculate Greeks only if requested
        if calculate_greeks:
            greeks = self._calculate_greeks('put')
            self.delta = greeks['delta']
            self.gamma = greeks['gamma']
            self.vega = greeks['vega']
            self.theta = greeks['theta']
            self.rho = greeks['rho']
        else:
            # Set Greeks to None when not calculated
            self.delta = None
            self.gamma = None
            self.vega = None
            self.theta = None
            self.rho = None


def compare_american_european(
    asset_price: float,
    strike_price: float,
    volatility: float,
    expiration_date: datetime.date,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: str = 'call'
) -> dict:
    """
    Compare American and European option prices.

    Args:
        asset_price: Current asset price
        strike_price: Strike price
        volatility: Volatility
        expiration_date: Expiration date
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield
        option_type: 'call' or 'put'

    Returns:
        Dictionary with comparison metrics
    """
    from options.euro_option_analysis import EuropeanCall, EuropeanPut

    if option_type == 'call':
        american = AmericanCall(asset_price, strike_price, volatility, expiration_date,
                               risk_free_rate, dividend_yield)
        european = EuropeanCall(asset_price, strike_price, volatility, expiration_date,
                               risk_free_rate, 0.0)  # European uses drift, not dividend
    else:
        american = AmericanPut(asset_price, strike_price, volatility, expiration_date,
                              risk_free_rate, dividend_yield)
        european = EuropeanPut(asset_price, strike_price, volatility, expiration_date,
                              risk_free_rate, 0.0)

    early_exercise_premium = american.price - european.price

    return {
        'american_price': american.price,
        'european_price': european.price,
        'early_exercise_premium': early_exercise_premium,
        'premium_percentage': (early_exercise_premium / european.price * 100) if european.price > 0 else 0
    }
