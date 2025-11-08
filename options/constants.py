"""
Constants Module

This module defines financial and computational constants used throughout
the algorithmic delta hedging library.
"""

# ==================== FINANCIAL CONSTANTS ====================

# Option Contract Specifications
OPTION_CONTRACT_MULTIPLIER = 100  # Standard US options contract size
OPTIONS_LOT_SIZE = 100  # Alias for contract multiplier

# Trading Calendar
TRADING_DAYS_PER_YEAR = 252  # Business days in a typical year
CALENDAR_DAYS_PER_YEAR = 365  # Calendar days per year
HOURS_PER_TRADING_DAY = 6.5  # Standard trading hours (9:30 AM - 4:00 PM EST)

# Market Parameters
DEFAULT_RISK_FREE_RATE = 0.04  # 4% annualized (typical US Treasury rate)
DEFAULT_DIVIDEND_YIELD = 0.0  # No dividends by default
DEFAULT_VOLATILITY = 0.20  # 20% annualized volatility
DEFAULT_DRIFT = 0.10  # 10% expected return

# ==================== NUMERICAL CONSTANTS ====================

# Computational Precision
EPSILON = 1e-10  # Small number to avoid division by zero
VEGA_MIN_THRESHOLD = 1e-10  # Minimum vega for implied volatility calculations
TOLERANCE_DEFAULT = 1e-6  # Default convergence tolerance

# Iteration Limits
MAX_ITERATIONS_DEFAULT = 100  # Default maximum iterations for numerical methods
MAX_ITERATIONS_IV = 100  # Maximum iterations for implied volatility
MAX_ITERATIONS_OPTIMIZATION = 1000  # Maximum iterations for optimization

# Binomial Tree Parameters
DEFAULT_BINOMIAL_STEPS = 100  # Default number of steps in binomial tree
MIN_BINOMIAL_STEPS = 10  # Minimum steps for reasonable accuracy
MAX_BINOMIAL_STEPS = 1000  # Maximum steps (performance consideration)

# Monte Carlo Simulation
DEFAULT_MC_SIMULATIONS = 10000  # Default number of Monte Carlo paths
MIN_MC_SIMULATIONS = 1000  # Minimum for statistical significance
MAX_MC_SIMULATIONS = 1000000  # Maximum (memory consideration)

# ==================== NUMERICAL METHODS ====================

# Finite Difference Step Sizes
DELTA_SPOT_PCT = 0.01  # 1% change in spot for delta calculation
DELTA_VOL_PCT = 0.01  # 1% change in volatility for vega calculation
DELTA_RATE_ABS = 0.01  # 1% absolute change in rate for rho calculation
DELTA_TIME_DAYS = 1  # 1 day for theta calculation

# ==================== RISK PARAMETERS ====================

# Value at Risk
VAR_CONFIDENCE_95 = 0.95  # 95% confidence level
VAR_CONFIDENCE_99 = 0.99  # 99% confidence level
VAR_LOOKBACK_DAYS = 252  # One year of historical data

# Position Limits
MAX_POSITION_SIZE_DEFAULT = 100000  # $100,000 default max position
MAX_PORTFOLIO_LEVERAGE = 2.0  # 2x leverage maximum
MAX_DELTA_EXPOSURE = 100000  # Maximum delta exposure

# ==================== DISPLAY PRECISION ====================

# Decimal Places for Display
PRICE_DECIMALS = 2  # Display prices to 2 decimal places
GREEK_DECIMALS = 4  # Display Greeks to 4 decimal places
PERCENTAGE_DECIMALS = 2  # Display percentages to 2 decimal places
VOLATILITY_DECIMALS = 4  # Display volatility to 4 decimal places

# ==================== VALIDATION BOUNDS ====================

# Price Bounds
MIN_ASSET_PRICE = 0.01  # Minimum valid asset price
MAX_ASSET_PRICE = 1000000.0  # Maximum valid asset price
MIN_STRIKE_PRICE = 0.01  # Minimum valid strike price
MAX_STRIKE_PRICE = 1000000.0  # Maximum valid strike price

# Volatility Bounds
MIN_VOLATILITY = 0.001  # 0.1% minimum volatility
MAX_VOLATILITY = 5.0  # 500% maximum volatility (extreme cases)
TYPICAL_MIN_VOLATILITY = 0.05  # 5% typical minimum
TYPICAL_MAX_VOLATILITY = 2.0  # 200% typical maximum

# Rate Bounds
MIN_RISK_FREE_RATE = -0.10  # -10% (negative rates possible)
MAX_RISK_FREE_RATE = 0.20  # 20% (extreme high rate environment)

# Time Bounds
MIN_TIME_TO_EXPIRATION = 1 / TRADING_DAYS_PER_YEAR  # 1 trading day
MAX_TIME_TO_EXPIRATION_YEARS = 10.0  # 10 years maximum

# ==================== STRING CONSTANTS ====================

# Option Types
CALL_OPTION = 'call'
PUT_OPTION = 'put'

# Position Types
LONG_POSITION = 'long'
SHORT_POSITION = 'short'

# Action Types
BUY_ACTION = 'buy'
SELL_ACTION = 'sell'

# ==================== HELPER FUNCTIONS ====================

def validate_price(price: float, name: str = "Price") -> None:
    """Validate that a price is within acceptable bounds."""
    if price is None:
        raise ValueError(f"{name} cannot be None")
    if price < MIN_ASSET_PRICE or price > MAX_ASSET_PRICE:
        raise ValueError(
            f"{name} must be between ${MIN_ASSET_PRICE} and ${MAX_ASSET_PRICE}, "
            f"got ${price}"
        )


def validate_volatility(volatility: float, allow_extreme: bool = False) -> None:
    """Validate that volatility is within acceptable bounds."""
    if volatility is None:
        raise ValueError("Volatility cannot be None")

    min_vol = MIN_VOLATILITY if allow_extreme else TYPICAL_MIN_VOLATILITY
    max_vol = MAX_VOLATILITY if allow_extreme else TYPICAL_MAX_VOLATILITY

    if volatility < min_vol or volatility > max_vol:
        raise ValueError(
            f"Volatility must be between {min_vol} and {max_vol}, "
            f"got {volatility}"
        )


def validate_time_to_expiration(time_years: float) -> None:
    """Validate that time to expiration is within acceptable bounds."""
    if time_years is None:
        raise ValueError("Time to expiration cannot be None")
    if time_years < 0:
        raise ValueError(f"Time to expiration cannot be negative, got {time_years}")
    if time_years > MAX_TIME_TO_EXPIRATION_YEARS:
        raise ValueError(
            f"Time to expiration cannot exceed {MAX_TIME_TO_EXPIRATION_YEARS} years, "
            f"got {time_years}"
        )


def years_to_trading_days(years: float) -> int:
    """Convert years to trading days."""
    return int(years * TRADING_DAYS_PER_YEAR)


def trading_days_to_years(days: int) -> float:
    """Convert trading days to years."""
    return days / TRADING_DAYS_PER_YEAR
