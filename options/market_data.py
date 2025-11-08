"""
Market Data Integration Module

This module provides integration with real market data sources including
stock prices, options chains, and historical data using yfinance and other APIs.
"""

import datetime
from datetime import timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketDataError(Exception):
    """Raised when market data retrieval fails."""
    pass


class MarketDataProvider:
    """
    Base class for market data providers.

    Provides a common interface for different data sources.
    """

    def get_stock_price(self, symbol: str) -> float:
        """Get current stock price."""
        raise NotImplementedError

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Get historical price data."""
        raise NotImplementedError

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime.date] = None
    ) -> Dict:
        """Get options chain data."""
        raise NotImplementedError

    def get_dividend_yield(self, symbol: str) -> float:
        """Get annual dividend yield."""
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    """
    Market data provider using yfinance library.

    Yahoo Finance provides free access to stock and options data.
    """

    def __init__(self):
        """Initialize YFinance provider."""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError(
                "yfinance is required for this provider. "
                "Install it with: pip install yfinance"
            )

    def get_stock_price(self, symbol: str) -> float:
        """
        Get current stock price.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current stock price

        Raises:
            MarketDataError: If data retrieval fails
        """
        try:
            ticker = self.yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if data.empty:
                raise MarketDataError(f"No data available for {symbol}")
            return float(data['Close'].iloc[-1])
        except Exception as e:
            raise MarketDataError(f"Failed to get stock price for {symbol}: {str(e)}")

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval ('1d', '1h', '1m', etc.)

        Returns:
            DataFrame with OHLCV data

        Raises:
            MarketDataError: If data retrieval fails
        """
        try:
            ticker = self.yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            if data.empty:
                raise MarketDataError(
                    f"No historical data available for {symbol} "
                    f"from {start_date} to {end_date}"
                )
            return data
        except Exception as e:
            raise MarketDataError(
                f"Failed to get historical data for {symbol}: {str(e)}"
            )

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime.date] = None
    ) -> Dict:
        """
        Get options chain data.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Specific expiration date (optional)

        Returns:
            Dictionary with calls and puts DataFrames

        Raises:
            MarketDataError: If data retrieval fails
        """
        try:
            ticker = self.yf.Ticker(symbol)

            # Get available expiration dates
            if not hasattr(ticker, 'options') or not ticker.options:
                raise MarketDataError(f"No options available for {symbol}")

            # Use specified expiration or nearest
            if expiration_date:
                exp_str = expiration_date.strftime('%Y-%m-%d')
                if exp_str not in ticker.options:
                    raise MarketDataError(
                        f"No options available for {symbol} on {exp_str}"
                    )
            else:
                exp_str = ticker.options[0]

            # Get options chain
            opt_chain = ticker.option_chain(exp_str)

            return {
                'calls': opt_chain.calls,
                'puts': opt_chain.puts,
                'expiration': exp_str,
                'underlying_price': self.get_stock_price(symbol)
            }
        except Exception as e:
            raise MarketDataError(
                f"Failed to get options chain for {symbol}: {str(e)}"
            )

    def get_dividend_yield(self, symbol: str) -> float:
        """
        Get annual dividend yield.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Annual dividend yield (as decimal, e.g., 0.02 for 2%)

        Raises:
            MarketDataError: If data retrieval fails
        """
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info

            # Try to get dividend yield
            if 'dividendYield' in info and info['dividendYield']:
                return float(info['dividendYield'])

            # Alternative: calculate from dividends
            if 'trailingAnnualDividendRate' in info and 'currentPrice' in info:
                div_rate = info.get('trailingAnnualDividendRate', 0)
                price = info.get('currentPrice', 1)
                if price > 0:
                    return div_rate / price

            # No dividend
            return 0.0
        except Exception as e:
            # If we can't get dividend info, assume no dividend
            return 0.0

    def get_all_expirations(self, symbol: str) -> List[str]:
        """
        Get all available option expiration dates.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of expiration date strings
        """
        try:
            ticker = self.yf.Ticker(symbol)
            return list(ticker.options)
        except Exception as e:
            raise MarketDataError(
                f"Failed to get expiration dates for {symbol}: {str(e)}"
            )

    def calculate_historical_volatility(
        self,
        symbol: str,
        window: int = 30,
        annualize: bool = True
    ) -> float:
        """
        Calculate historical volatility from price data.

        Args:
            symbol: Stock ticker symbol
            window: Number of days for calculation
            annualize: Whether to annualize the volatility

        Returns:
            Historical volatility

        Raises:
            MarketDataError: If calculation fails
        """
        try:
            # Get historical data
            end_date = datetime.date.today()
            start_date = end_date - timedelta(days=window + 10)  # Extra buffer

            data = self.get_historical_data(symbol, start_date, end_date)

            # Calculate log returns
            data['returns'] = np.log(data['Close'] / data['Close'].shift(1))

            # Calculate volatility (standard deviation of returns)
            vol = data['returns'].std()

            # Annualize if requested
            if annualize:
                vol = vol * np.sqrt(252)  # 252 trading days per year

            return float(vol)
        except Exception as e:
            raise MarketDataError(
                f"Failed to calculate historical volatility for {symbol}: {str(e)}"
            )


class DataCache:
    """
    Simple cache for market data to reduce API calls.
    """

    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached data in seconds
        """
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[any]:
        """
        Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.datetime.now() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: any) -> None:
        """
        Set cache value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, datetime.datetime.now())

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()


class CachedMarketDataProvider(YFinanceProvider):
    """
    Market data provider with caching to reduce API calls.
    """

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize cached provider.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__()
        self.cache = DataCache(cache_ttl)

    def get_stock_price(self, symbol: str) -> float:
        """Get current stock price with caching."""
        key = f"price_{symbol}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        price = super().get_stock_price(symbol)
        self.cache.set(key, price)
        return price

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime.date] = None
    ) -> Dict:
        """Get options chain with caching."""
        exp_str = expiration_date.strftime('%Y-%m-%d') if expiration_date else 'nearest'
        key = f"options_{symbol}_{exp_str}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        chain = super().get_options_chain(symbol, expiration_date)
        self.cache.set(key, chain)
        return chain


def get_risk_free_rate() -> float:
    """
    Get current risk-free rate (US 10-year Treasury).

    Returns:
        Risk-free rate as decimal

    Note:
        This is a simplified implementation. In production, you would
        fetch this from a reliable data source.
    """
    try:
        import yfinance as yf
        # US 10-year Treasury
        treasury = yf.Ticker("^TNX")
        data = treasury.history(period='1d')
        if not data.empty:
            # TNX is in percentage, convert to decimal
            return float(data['Close'].iloc[-1]) / 100
    except Exception:
        pass

    # Default to 4% if unable to fetch
    return 0.04


def get_market_data_summary(symbol: str, provider: Optional[MarketDataProvider] = None) -> Dict:
    """
    Get comprehensive market data summary for a symbol.

    Args:
        symbol: Stock ticker symbol
        provider: Market data provider (default: YFinanceProvider)

    Returns:
        Dictionary with market data summary
    """
    if provider is None:
        provider = YFinanceProvider()

    try:
        price = provider.get_stock_price(symbol)
        dividend_yield = provider.get_dividend_yield(symbol)
        hist_vol = provider.calculate_historical_volatility(symbol, window=30)

        # Get available expirations
        if isinstance(provider, YFinanceProvider):
            expirations = provider.get_all_expirations(symbol)
        else:
            expirations = []

        return {
            'symbol': symbol,
            'current_price': price,
            'dividend_yield': dividend_yield,
            'historical_volatility_30d': hist_vol,
            'available_expirations': expirations,
            'risk_free_rate': get_risk_free_rate(),
            'timestamp': datetime.datetime.now()
        }
    except MarketDataError as e:
        raise e
    except Exception as e:
        raise MarketDataError(f"Failed to get market summary for {symbol}: {str(e)}")
