"""
Implied Volatility Calculation Module

This module implements implied volatility calculation using various numerical methods
including Newton-Raphson, bisection, and Brent's method.
"""

import math
import datetime
from typing import Optional, Callable
import numpy as np
from scipy import stats
from scipy.optimize import brentq, newton


class ImpliedVolatilityError(Exception):
    """Raised when implied volatility calculation fails."""
    pass


def black_scholes_price(
    asset_price: float,
    strike_price: float,
    volatility: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        asset_price: Current asset price
        strike_price: Strike price
        volatility: Volatility (annualized)
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if time_to_expiration <= 0:
        if option_type == 'call':
            return max(0, asset_price - strike_price)
        else:
            return max(0, strike_price - asset_price)

    d1 = (math.log(asset_price / strike_price) +
          (risk_free_rate + 0.5 * volatility**2) * time_to_expiration) / \
         (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)

    if option_type == 'call':
        price = asset_price * stats.norm.cdf(d1) - \
                strike_price * math.exp(-risk_free_rate * time_to_expiration) * stats.norm.cdf(d2)
    else:  # put
        price = strike_price * math.exp(-risk_free_rate * time_to_expiration) * stats.norm.cdf(-d2) - \
                asset_price * stats.norm.cdf(-d1)

    return price


def black_scholes_vega(
    asset_price: float,
    strike_price: float,
    volatility: float,
    time_to_expiration: float,
    risk_free_rate: float
) -> float:
    """
    Calculate vega (derivative of price with respect to volatility).

    Args:
        asset_price: Current asset price
        strike_price: Strike price
        volatility: Volatility (annualized)
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Vega value
    """
    if time_to_expiration <= 0 or volatility <= 0:
        return 0.0

    d1 = (math.log(asset_price / strike_price) +
          (risk_free_rate + 0.5 * volatility**2) * time_to_expiration) / \
         (volatility * math.sqrt(time_to_expiration))

    vega = asset_price * stats.norm.pdf(d1) * math.sqrt(time_to_expiration)
    return vega


def implied_volatility_newton_raphson(
    option_price: float,
    asset_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call',
    initial_guess: float = 0.3,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    This is the most commonly used method for implied volatility calculation.
    It uses the derivative (vega) to converge quickly.

    Args:
        option_price: Observed market price of the option
        asset_price: Current asset price
        strike_price: Strike price
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess (default 30%)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        Implied volatility

    Raises:
        ImpliedVolatilityError: If calculation fails to converge
    """
    # Input validation
    if option_price <= 0:
        raise ImpliedVolatilityError("Option price must be positive")
    if time_to_expiration <= 0:
        raise ImpliedVolatilityError("Time to expiration must be positive")

    # Check for arbitrage bounds
    if option_type == 'call':
        intrinsic_value = max(0, asset_price - strike_price)
        max_value = asset_price
    else:
        intrinsic_value = max(0, strike_price - asset_price)
        max_value = strike_price

    if option_price < intrinsic_value:
        raise ImpliedVolatilityError("Option price below intrinsic value (arbitrage)")
    if option_price > max_value:
        raise ImpliedVolatilityError("Option price exceeds maximum theoretical value")

    # Newton-Raphson iteration
    volatility = initial_guess

    for i in range(max_iterations):
        # Calculate price and vega at current volatility
        calculated_price = black_scholes_price(
            asset_price, strike_price, volatility,
            time_to_expiration, risk_free_rate, option_type
        )

        vega = black_scholes_vega(
            asset_price, strike_price, volatility,
            time_to_expiration, risk_free_rate
        )

        # Price difference
        price_diff = calculated_price - option_price

        # Check convergence
        if abs(price_diff) < tolerance:
            return volatility

        # Avoid division by zero
        if vega < 1e-10:
            raise ImpliedVolatilityError("Vega too small, cannot converge")

        # Newton-Raphson update
        volatility = volatility - price_diff / vega

        # Ensure volatility stays positive
        if volatility <= 0:
            volatility = initial_guess / 2

    raise ImpliedVolatilityError(
        f"Failed to converge after {max_iterations} iterations"
    )


def implied_volatility_bisection(
    option_price: float,
    asset_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call',
    vol_min: float = 0.001,
    vol_max: float = 5.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> float:
    """
    Calculate implied volatility using bisection method.

    More robust than Newton-Raphson but slower. Guaranteed to converge
    if a solution exists in the search range.

    Args:
        option_price: Observed market price of the option
        asset_price: Current asset price
        strike_price: Strike price
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        vol_min: Minimum volatility bound
        vol_max: Maximum volatility bound
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        Implied volatility

    Raises:
        ImpliedVolatilityError: If calculation fails
    """
    def objective(vol):
        return black_scholes_price(
            asset_price, strike_price, vol,
            time_to_expiration, risk_free_rate, option_type
        ) - option_price

    # Check if solution exists in range
    f_min = objective(vol_min)
    f_max = objective(vol_max)

    if f_min * f_max > 0:
        raise ImpliedVolatilityError(
            "Solution not bracketed in volatility range"
        )

    # Bisection
    for i in range(max_iterations):
        vol_mid = (vol_min + vol_max) / 2
        f_mid = objective(vol_mid)

        if abs(f_mid) < tolerance or (vol_max - vol_min) / 2 < tolerance:
            return vol_mid

        if f_min * f_mid < 0:
            vol_max = vol_mid
            f_max = f_mid
        else:
            vol_min = vol_mid
            f_min = f_mid

    return (vol_min + vol_max) / 2


def implied_volatility_brent(
    option_price: float,
    asset_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call',
    vol_min: float = 0.001,
    vol_max: float = 5.0
) -> float:
    """
    Calculate implied volatility using Brent's method (scipy).

    Combines bisection, secant, and inverse quadratic interpolation.
    Generally the most efficient method.

    Args:
        option_price: Observed market price of the option
        asset_price: Current asset price
        strike_price: Strike price
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        vol_min: Minimum volatility bound
        vol_max: Maximum volatility bound

    Returns:
        Implied volatility

    Raises:
        ImpliedVolatilityError: If calculation fails
    """
    def objective(vol):
        return black_scholes_price(
            asset_price, strike_price, vol,
            time_to_expiration, risk_free_rate, option_type
        ) - option_price

    try:
        result = brentq(objective, vol_min, vol_max, xtol=1e-6)
        return result
    except ValueError as e:
        raise ImpliedVolatilityError(f"Brent's method failed: {str(e)}")


def implied_volatility(
    option_price: float,
    asset_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call',
    method: str = 'newton',
    **kwargs
) -> float:
    """
    Calculate implied volatility using specified method.

    This is the main interface for implied volatility calculation.

    Args:
        option_price: Observed market price of the option
        asset_price: Current asset price
        strike_price: Strike price
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        method: 'newton', 'bisection', or 'brent'
        **kwargs: Additional arguments for specific methods

    Returns:
        Implied volatility

    Raises:
        ImpliedVolatilityError: If calculation fails
        ValueError: If method is unknown

    Examples:
        >>> iv = implied_volatility(5.0, 100, 105, 0.25, 0.05, 'call')
        >>> iv = implied_volatility(3.0, 100, 95, 0.25, 0.05, 'put', method='brent')
    """
    if method == 'newton':
        return implied_volatility_newton_raphson(
            option_price, asset_price, strike_price,
            time_to_expiration, risk_free_rate, option_type, **kwargs
        )
    elif method == 'bisection':
        return implied_volatility_bisection(
            option_price, asset_price, strike_price,
            time_to_expiration, risk_free_rate, option_type, **kwargs
        )
    elif method == 'brent':
        return implied_volatility_brent(
            option_price, asset_price, strike_price,
            time_to_expiration, risk_free_rate, option_type, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'newton', 'bisection', or 'brent'")


def calculate_volatility_smile(
    option_prices: list,
    strike_prices: list,
    asset_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call'
) -> dict:
    """
    Calculate implied volatility smile from a chain of options.

    Args:
        option_prices: List of observed option prices
        strike_prices: List of corresponding strike prices
        asset_price: Current asset price
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate
        option_type: 'call' or 'put'

    Returns:
        Dictionary with strikes and implied volatilities
    """
    ivs = []
    valid_strikes = []

    for price, strike in zip(option_prices, strike_prices):
        try:
            iv = implied_volatility(
                price, asset_price, strike,
                time_to_expiration, risk_free_rate,
                option_type, method='brent'
            )
            ivs.append(iv)
            valid_strikes.append(strike)
        except ImpliedVolatilityError:
            # Skip options that don't have valid IV
            continue

    return {
        'strikes': valid_strikes,
        'implied_volatilities': ivs,
        'moneyness': [strike / asset_price for strike in valid_strikes]
    }


def calculate_volatility_surface(
    option_chain_data: dict,
    asset_price: float,
    risk_free_rate: float
) -> dict:
    """
    Calculate full implied volatility surface across strikes and expirations.

    Args:
        option_chain_data: Dictionary with structure:
            {
                'expiration1': {'strikes': [...], 'call_prices': [...], 'put_prices': [...]},
                'expiration2': ...
            }
        asset_price: Current asset price
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with volatility surface data
    """
    surface_data = {
        'expirations': [],
        'strikes': [],
        'call_ivs': [],
        'put_ivs': [],
        'moneyness': [],
        'time_to_expiration': []
    }

    for expiration, data in option_chain_data.items():
        if isinstance(expiration, datetime.date):
            dte = np.busday_count(datetime.date.today(), expiration) / 252
        else:
            dte = expiration  # Assume it's already in years

        strikes = data['strikes']
        call_prices = data.get('call_prices', [])
        put_prices = data.get('put_prices', [])

        # Calculate IV for calls
        for strike, call_price in zip(strikes, call_prices):
            try:
                iv = implied_volatility(call_price, asset_price, strike, dte,
                                      risk_free_rate, 'call', method='brent')
                surface_data['expirations'].append(expiration)
                surface_data['strikes'].append(strike)
                surface_data['call_ivs'].append(iv)
                surface_data['moneyness'].append(strike / asset_price)
                surface_data['time_to_expiration'].append(dte)
            except (ImpliedVolatilityError, ValueError):
                continue

        # Calculate IV for puts
        for strike, put_price in zip(strikes, put_prices):
            try:
                iv = implied_volatility(put_price, asset_price, strike, dte,
                                      risk_free_rate, 'put', method='brent')
                if strike not in surface_data['strikes'] or \
                   expiration not in surface_data['expirations']:
                    surface_data['put_ivs'].append(iv)
            except (ImpliedVolatilityError, ValueError):
                continue

    return surface_data
