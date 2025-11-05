# Algorithmic Delta Hedging

A production-ready Python library for options pricing, delta hedging strategies, portfolio management, and quantitative finance research.

## Features Overview

### Options Pricing Models

#### European Options
- Black-Scholes pricing with all Greeks (Delta, Gamma, Vega, Theta, Rho)
- Real-time visualization capabilities
- Dividend-adjusted models (continuous and discrete)

#### American Options
- Binomial tree pricing method
- Early exercise optimization
- Complete Greeks calculation
- Comparison with European pricing

#### Exotic Options
- **Asian Options**: Average price/strike options with Monte Carlo pricing
- **Barrier Options**: Knock-in/knock-out options
- **Lookback Options**: Min/max price dependent payoffs
- **Digital Options**: Binary payoff options

### Implied Volatility
- **Multiple Methods**: Newton-Raphson, Bisection, Brent's method
- **Volatility Surface**: Full IV surface calculation across strikes and maturities
- **Volatility Smile**: Extract and analyze market-implied volatility patterns

### Options Strategies
- **Spreads**: Bull/Bear Call/Put spreads
- **Straddles & Strangles**: Long/Short positions
- **Iron Condors**: Multi-leg range-bound strategies
- **Butterflies**: Symmetric volatility plays
- **Covered Calls & Protective Puts**
- **Custom Strategy Builder**: Create any multi-leg strategy

### Delta Hedging & Portfolio Management
- **Delta-Neutral Hedging**: Automatic rebalancing based on thresholds
- **Transaction Costs**: Commission and slippage modeling
- **Portfolio Tracking**: Multi-asset position management
- **P&L Attribution**: Real-time profit/loss tracking
- **Greeks Aggregation**: Portfolio-level Greeks calculation

### Risk Management
- **Value at Risk (VaR)**: Historical, Parametric, and Monte Carlo methods
- **Conditional VaR (CVaR)**: Expected shortfall calculations
- **Maximum Drawdown**: Peak-to-trough analysis
- **Sharpe & Sortino Ratios**: Risk-adjusted performance metrics
- **Stress Testing**: Scenario-based risk analysis
- **Beta Calculation**: Systematic risk measurement

### Market Data Integration
- **Real-Time Data**: yfinance integration for stocks and options
- **Historical Data**: OHLCV data with configurable intervals
- **Options Chains**: Live options market data
- **Dividend Information**: Automatic dividend yield extraction
- **Data Caching**: Efficient API call management

### Backtesting Framework
- **Strategy Testing**: Backtest any options strategy on historical data
- **Performance Metrics**: Comprehensive performance analysis
- **Equity Curves**: Visual representation of strategy performance
- **Commission Modeling**: Realistic transaction costs
- **Customizable**: Easy to implement custom strategies

### Machine Learning
- **GARCH Models**: Volatility clustering and forecasting
- **LSTM Networks**: Deep learning for volatility prediction
- **Random Forest**: Ensemble methods for vol forecasting
- **Ensemble Models**: Combine multiple forecasters

### Web Dashboard
- **Interactive UI**: Streamlit-based web interface
- **Options Pricing Calculator**: Real-time pricing with parameter controls
- **Greeks Visualization**: Interactive Greeks sensitivity analysis
- **Strategy Builder**: Visual strategy construction
- **Portfolio Manager**: Live portfolio tracking
- **Risk Analytics**: Real-time risk metrics dashboard
- **Market Data Viewer**: Live market data integration

### Performance Optimization
- **Parallel Computing**: Multi-threaded Monte Carlo simulations
- **Vectorization**: NumPy-optimized calculations
- **Efficient Algorithms**: Optimized numerical methods
- **Benchmarking Tools**: Performance comparison utilities

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ScientiaCapital/Algorithmic_Delta_Hedging.git
cd Algorithmic_Delta_Hedging

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Options Pricing

```python
import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut

# Create an option
expiration = datetime.date.today() + timedelta(days=30)
call = EuropeanCall(
    asset_price=100.0,
    strike_price=105.0,
    volatility=0.30,
    expiration_date=expiration,
    risk_free_rate=0.05,
    drift=0.10
)

print(f"Price: ${call.price:.2f}")
print(f"Delta: {call.delta:.4f}")
print(f"Gamma: {call.gamma:.4f}")
print(f"Vega: {call.vega:.4f}")
print(f"Theta: {call.theta:.4f}")
print(f"Rho: {call.rho:.4f}")
```

### Web Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run streamlit_app/dashboard.py
```

Then open your browser to `http://localhost:8501`

For complete documentation and advanced examples, see the full README and examples directory.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=options --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Not for actual trading without proper validation and risk management.

---

**For complete documentation, examples, and API reference, see the docs directory and examples folder.**
