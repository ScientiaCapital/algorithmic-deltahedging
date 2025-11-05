"""
Dividend-Adjusted Options Pricing Module

This module implements Black-Scholes pricing with dividend adjustments
for options on dividend-paying stocks.
"""

import math
import datetime
from typing import Optional
import numpy as np
from scipy import stats


class DividendAdjustedCall:
    """
    European Call Option with continuous dividend yield adjustment.

    Uses the Black-Scholes-Merton model which accounts for continuous dividends.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        expiration_date: datetime.date,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        """
        Initialize dividend-adjusted call option.

        Args:
            asset_price: Current price of the underlying asset
            strike_price: Strike price of the option
            volatility: Annualized volatility
            expiration_date: Expiration date
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual continuous dividend yield

        Raises:
            ValueError: If any parameters are invalid
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
        self.dividend_yield = dividend_yield

        # Calculate time to expiration
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        self.dt = max(dt, 0)

        if self.dt > 0:
            # Calculate d1 and d2 with dividend adjustment
            d1 = self._calculate_d1()
            d2 = self._calculate_d2(d1)

            # Calculate option price and Greeks
            self.price = self._calculate_price(d1, d2)
            self.delta = self._calculate_delta(d1)
            self.gamma = self._calculate_gamma(d1)
            self.vega = self._calculate_vega(d1)
            self.theta = self._calculate_theta(d1, d2)
            self.rho = self._calculate_rho(d2)
        else:
            # Option has expired
            self.price = max(0, asset_price - strike_price)
            self.delta = 1.0 if asset_price > strike_price else 0.0
            self.gamma = 0.0
            self.vega = 0.0
            self.theta = 0.0
            self.rho = 0.0

    def _calculate_d1(self) -> float:
        """Calculate d1 parameter with dividend adjustment."""
        numerator = (math.log(self.asset_price / self.strike_price) +
                    (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility**2) * self.dt)
        denominator = self.volatility * math.sqrt(self.dt)
        return numerator / denominator

    def _calculate_d2(self, d1: float) -> float:
        """Calculate d2 parameter."""
        return d1 - self.volatility * math.sqrt(self.dt)

    def _calculate_price(self, d1: float, d2: float) -> float:
        """Calculate call option price with dividend adjustment."""
        # Adjust asset price for dividends
        adjusted_asset_price = self.asset_price * math.exp(-self.dividend_yield * self.dt)

        n1 = stats.norm.cdf(d1)
        n2 = stats.norm.cdf(d2)

        price = (adjusted_asset_price * math.exp(self.dividend_yield * self.dt) * n1 -
                self.strike_price * math.exp(-self.risk_free_rate * self.dt) * n2)
        return price

    def _calculate_delta(self, d1: float) -> float:
        """Calculate delta with dividend adjustment."""
        return math.exp(-self.dividend_yield * self.dt) * stats.norm.cdf(d1)

    def _calculate_gamma(self, d1: float) -> float:
        """Calculate gamma with dividend adjustment."""
        numerator = math.exp(-self.dividend_yield * self.dt) * stats.norm.pdf(d1)
        denominator = self.asset_price * self.volatility * math.sqrt(self.dt)
        return numerator / denominator

    def _calculate_vega(self, d1: float) -> float:
        """Calculate vega with dividend adjustment."""
        vega = (self.asset_price * math.exp(-self.dividend_yield * self.dt) *
                stats.norm.pdf(d1) * math.sqrt(self.dt))
        return vega / 100  # Per 1% change

    def _calculate_theta(self, d1: float, d2: float) -> float:
        """Calculate theta with dividend adjustment."""
        term1 = -(self.asset_price * stats.norm.pdf(d1) * self.volatility *
                 math.exp(-self.dividend_yield * self.dt)) / (2 * math.sqrt(self.dt))

        term2 = (self.dividend_yield * self.asset_price *
                stats.norm.cdf(d1) * math.exp(-self.dividend_yield * self.dt))

        term3 = (self.risk_free_rate * self.strike_price *
                math.exp(-self.risk_free_rate * self.dt) * stats.norm.cdf(d2))

        theta = (term1 + term2 - term3) / 365  # Per day
        return theta

    def _calculate_rho(self, d2: float) -> float:
        """Calculate rho with dividend adjustment."""
        rho = (self.strike_price * self.dt *
              math.exp(-self.risk_free_rate * self.dt) * stats.norm.cdf(d2))
        return rho / 100  # Per 1% change


class DividendAdjustedPut:
    """
    European Put Option with continuous dividend yield adjustment.

    Uses the Black-Scholes-Merton model which accounts for continuous dividends.
    """

    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        volatility: float,
        expiration_date: datetime.date,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        """
        Initialize dividend-adjusted put option.

        Args:
            asset_price: Current price of the underlying asset
            strike_price: Strike price of the option
            volatility: Annualized volatility
            expiration_date: Expiration date
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual continuous dividend yield

        Raises:
            ValueError: If any parameters are invalid
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
        self.dividend_yield = dividend_yield

        # Calculate time to expiration
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        self.dt = max(dt, 0)

        if self.dt > 0:
            # Calculate d1 and d2 with dividend adjustment
            d1 = self._calculate_d1()
            d2 = self._calculate_d2(d1)

            # Calculate option price and Greeks
            self.price = self._calculate_price(d1, d2)
            self.delta = self._calculate_delta(d1)
            self.gamma = self._calculate_gamma(d1)
            self.vega = self._calculate_vega(d1)
            self.theta = self._calculate_theta(d1, d2)
            self.rho = self._calculate_rho(d2)
        else:
            # Option has expired
            self.price = max(0, strike_price - asset_price)
            self.delta = -1.0 if asset_price < strike_price else 0.0
            self.gamma = 0.0
            self.vega = 0.0
            self.theta = 0.0
            self.rho = 0.0

    def _calculate_d1(self) -> float:
        """Calculate d1 parameter with dividend adjustment."""
        numerator = (math.log(self.asset_price / self.strike_price) +
                    (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility**2) * self.dt)
        denominator = self.volatility * math.sqrt(self.dt)
        return numerator / denominator

    def _calculate_d2(self, d1: float) -> float:
        """Calculate d2 parameter."""
        return d1 - self.volatility * math.sqrt(self.dt)

    def _calculate_price(self, d1: float, d2: float) -> float:
        """Calculate put option price with dividend adjustment."""
        n1 = stats.norm.cdf(-d1)
        n2 = stats.norm.cdf(-d2)

        price = (self.strike_price * math.exp(-self.risk_free_rate * self.dt) * n2 -
                self.asset_price * math.exp(-self.dividend_yield * self.dt) * n1)
        return price

    def _calculate_delta(self, d1: float) -> float:
        """Calculate delta with dividend adjustment."""
        return -math.exp(-self.dividend_yield * self.dt) * stats.norm.cdf(-d1)

    def _calculate_gamma(self, d1: float) -> float:
        """Calculate gamma with dividend adjustment."""
        numerator = math.exp(-self.dividend_yield * self.dt) * stats.norm.pdf(d1)
        denominator = self.asset_price * self.volatility * math.sqrt(self.dt)
        return numerator / denominator

    def _calculate_vega(self, d1: float) -> float:
        """Calculate vega with dividend adjustment."""
        vega = (self.asset_price * math.exp(-self.dividend_yield * self.dt) *
                stats.norm.pdf(d1) * math.sqrt(self.dt))
        return vega / 100  # Per 1% change

    def _calculate_theta(self, d1: float, d2: float) -> float:
        """Calculate theta with dividend adjustment."""
        term1 = -(self.asset_price * stats.norm.pdf(d1) * self.volatility *
                 math.exp(-self.dividend_yield * self.dt)) / (2 * math.sqrt(self.dt))

        term2 = (self.dividend_yield * self.asset_price *
                stats.norm.cdf(-d1) * math.exp(-self.dividend_yield * self.dt))

        term3 = (self.risk_free_rate * self.strike_price *
                math.exp(-self.risk_free_rate * self.dt) * stats.norm.cdf(-d2))

        theta = (term1 - term2 + term3) / 365  # Per day
        return theta

    def _calculate_rho(self, d2: float) -> float:
        """Calculate rho with dividend adjustment."""
        rho = -(self.strike_price * self.dt *
               math.exp(-self.risk_free_rate * self.dt) * stats.norm.cdf(-d2))
        return rho / 100  # Per 1% change


class DiscreteDividendOption:
    """
    Option pricing with discrete dividend payments.

    This model adjusts for known future dividend payments.
    """

    @staticmethod
    def adjust_spot_price(
        spot_price: float,
        dividend_dates: list,
        dividend_amounts: list,
        risk_free_rate: float,
        valuation_date: datetime.date
    ) -> float:
        """
        Adjust spot price for present value of known dividends.

        Args:
            spot_price: Current spot price
            dividend_dates: List of dividend payment dates
            dividend_amounts: List of dividend amounts
            risk_free_rate: Risk-free rate
            valuation_date: Current date

        Returns:
            Adjusted spot price
        """
        pv_dividends = 0.0

        for div_date, div_amount in zip(dividend_dates, dividend_amounts):
            if div_date > valuation_date:
                # Calculate time to dividend in years
                days_to_div = (div_date - valuation_date).days
                years_to_div = days_to_div / 365.0

                # Present value of dividend
                pv = div_amount * math.exp(-risk_free_rate * years_to_div)
                pv_dividends += pv

        # Adjusted spot price
        adjusted_spot = spot_price - pv_dividends
        return max(adjusted_spot, 0.01)  # Ensure positive

    @staticmethod
    def price_with_discrete_dividends(
        asset_price: float,
        strike_price: float,
        volatility: float,
        time_to_expiration: float,
        risk_free_rate: float,
        dividend_dates: list,
        dividend_amounts: list,
        option_type: str = 'call'
    ) -> float:
        """
        Price option with discrete dividends.

        Args:
            asset_price: Current asset price
            strike_price: Strike price
            volatility: Volatility
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate
            dividend_dates: List of dividend dates
            dividend_amounts: List of dividend amounts
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        from options.euro_option_analysis import EuropeanCall, EuropeanPut

        # Adjust spot price for dividends
        valuation_date = datetime.date.today()
        adjusted_price = DiscreteDividendOption.adjust_spot_price(
            asset_price, dividend_dates, dividend_amounts,
            risk_free_rate, valuation_date
        )

        # Calculate expiration date from time to expiration
        expiration_date = valuation_date + datetime.timedelta(
            days=int(time_to_expiration * 365)
        )

        # Price with adjusted spot
        if option_type == 'call':
            option = EuropeanCall(
                adjusted_price, strike_price, volatility,
                expiration_date, risk_free_rate, 0.0
            )
        else:
            option = EuropeanPut(
                adjusted_price, strike_price, volatility,
                expiration_date, risk_free_rate, 0.0
            )

        return option.price
