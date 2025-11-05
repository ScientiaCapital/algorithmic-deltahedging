# Algorithmic Delta Hedging

A comprehensive Python library for pricing, analyzing, and implementing delta hedging strategies for European options using the Black-Scholes model.

## Features

### Current Implementation
- **European Call & Put Options**: Complete Black-Scholes pricing implementation
- **Option Greeks**: Delta calculations with real-time tracking
- **Visualization**: Live animated plots showing option price, delta, and underlying asset movements
- **Geometric Brownian Motion**: Proper GBM simulation for asset price movements
- **Type Safety**: Full type hints and comprehensive docstrings
- **Input Validation**: Robust error handling for all parameters

### In Development
- Additional Greeks (Gamma, Vega, Theta, Rho)
- Delta hedging strategy implementation
- Portfolio management system
- Transaction cost modeling
- Backtesting framework
- Real market data integration

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ScientiaCapital/Algorithmic_Delta_Hedging.git
cd Algorithmic_Delta_Hedging
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut

# Calculate expiration date (30 days from today)
expiration = datetime.date.today() + timedelta(days=30)

# Create a European call option
call_option = EuropeanCall(
    asset_price=100.0,      # Current stock price
    strike_price=105.0,     # Strike price
    volatility=0.3,         # 30% annual volatility
    expiration_date=expiration,
    risk_free_rate=0.05,    # 5% risk-free rate
    drift=0.1               # 10% expected return
)

# Access option properties
print(f"Option Price: ${call_option.price:.2f}")
print(f"Delta: {call_option.delta:.4f}")
print(f"Exercise Probability: {call_option.exercise_prob():.2%}")
```

### Put Options

```python
# Create a European put option
put_option = EuropeanPut(
    asset_price=100.0,
    strike_price=95.0,
    volatility=0.3,
    expiration_date=expiration,
    risk_free_rate=0.05,
    drift=0.1
)

print(f"Put Price: ${put_option.price:.2f}")
print(f"Put Delta: {put_option.delta:.4f}")
```

### Live Visualization

```python
from options.euro_option_analysis import LiveOptionsGraph

# Create option
call_option = EuropeanCall(100, 105, 0.3, expiration, 0.05, 0.1)

# Launch live visualization (displays animated plots)
graph = LiveOptionsGraph(call_option, 'call')
```

The visualization shows three real-time plots:
1. **Option Price**: Black-Scholes theoretical price over time
2. **Delta**: Sensitivity to underlying price changes
3. **Asset Price**: Underlying asset price vs strike price (color-coded for ITM/OTM)

## Project Structure

```
Algorithmic_Delta_Hedging/
├── options/
│   └── euro_option_analysis.py   # European options implementation
├── tests/                         # Unit tests (coming soon)
├── examples/                      # Example scripts (coming soon)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## Mathematical Background

This library implements the Black-Scholes formula for European options:

**Call Option Price:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option Price:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
- `d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `S₀` = Current asset price
- `K` = Strike price
- `r` = Risk-free rate
- `σ` = Volatility
- `T` = Time to expiration
- `N(·)` = Cumulative standard normal distribution

**Delta (Call):** `Δ = N(d₁)`
**Delta (Put):** `Δ = N(d₁) - 1`

Asset prices are simulated using Geometric Brownian Motion:
```
S(t+dt) = S(t) × exp[(μ - σ²/2)dt + σ√dt·Z]
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=options --cov-report=html
```

## Development

### Code Quality

Format code with Black:
```bash
black options/
```

Check types with mypy:
```bash
mypy options/
```

Lint with flake8:
```bash
flake8 options/
```

## Roadmap

### Phase 1: Foundation (Completed)
- [x] European call/put pricing
- [x] Delta calculation
- [x] Real-time visualization
- [x] Type hints and documentation
- [x] Input validation

### Phase 2: Core Hedging (In Progress)
- [ ] Gamma, Vega, Theta, Rho calculations
- [ ] Delta hedging strategy class
- [ ] Portfolio management system
- [ ] Transaction cost modeling
- [ ] P&L tracking

### Phase 3: Testing & Infrastructure
- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] CI/CD pipeline
- [ ] Performance benchmarks

### Phase 4: Advanced Features
- [ ] American options
- [ ] Implied volatility calculation
- [ ] Options strategies (spreads, straddles, etc.)
- [ ] Real market data integration
- [ ] Backtesting framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used for actual trading without thorough testing and validation. The authors are not responsible for any financial losses incurred through the use of this software.

## References

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Hull, J. C. (2018). "Options, Futures, and Other Derivatives"
- Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance"

## Contact

For questions or feedback, please open an issue on GitHub.
