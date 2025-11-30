# PRP: <Feature Name>

**CRITICAL RULE**: NO OpenAI models allowed in this project. Use appropriate alternatives only.

## Metadata
- **PRP ID**: `prp_<feature-name>_<YYYYMMDD>`
- **Author**: <author-name>
- **Created**: <YYYY-MM-DD>
- **Status**: Draft | In Progress | Complete
- **Priority**: High | Medium | Low
- **Estimated Hours**: <hours>

## Feature Overview

### Objective
[Clear, concise description of what this feature accomplishes]

### Business Value
- **For Traders**: [How this helps options traders]
- **For Portfolio Managers**: [How this improves risk management]
- **For Analysts**: [How this enhances analysis capabilities]

### Success Metrics
- Pricing accuracy: ±X% vs theoretical values
- Performance: <Xms per calculation
- Test coverage: >80%
- User adoption: X% of dashboard users

## Technical Specification

### Options Pricing Models
**Primary Model**: [Black-Scholes | Binomial | Monte Carlo]

**Formulas**:
```
Call Price: C = S*N(d1) - K*e^(-rT)*N(d2)
Put Price: P = K*e^(-rT)*N(-d2) - S*N(-d1)

where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

**Assumptions**:
- Risk-free rate: [Source]
- Volatility estimation: [Historical | Implied]
- Dividend handling: [Continuous | Discrete]

### Greeks Calculations
| Greek | Formula | Accuracy Target |
|-------|---------|-----------------|
| Delta | ∂C/∂S = N(d1) | ±0.001 |
| Gamma | ∂²C/∂S² = N'(d1)/(S*σ√T) | ±0.0001 |
| Theta | ∂C/∂t | ±0.01 |
| Vega | ∂C/∂σ = S*N'(d1)√T | ±0.01 |
| Rho | ∂C/∂r = K*T*e^(-rT)*N(d2) | ±0.01 |

### Data Pipeline
**Data Sources**:
- Market prices: yfinance (SPY, QQQ, individual stocks)
- Risk-free rate: [Source - e.g., Fed Funds, Treasury yields]
- Volatility: [Calculated from historical data]

**Data Flow**:
```
yfinance API → DataFrame → NumPy arrays → Pricing calculations → Streamlit UI
```

**Error Handling**:
- Network failures: Retry with exponential backoff
- Missing data: Use last known value + warning
- Invalid inputs: Validation with clear error messages

### Implementation Architecture

#### Module Structure
```
options/
├── <feature>/
│   ├── __init__.py
│   ├── pricing.py          # Black-Scholes, binomial, MC
│   ├── greeks.py           # Delta, gamma, theta, vega, rho
│   ├── hedging.py          # Delta hedging, rebalancing
│   └── validation.py       # Input validation, bounds checking
```

#### Dependencies
```python
# Core scientific computing (NO OpenAI)
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Data fetching
yfinance>=0.2.0

# UI
streamlit>=1.25.0

# ML (if needed - NO OpenAI)
scikit-learn>=1.3.0  # For regression, clustering only
```

### Performance Requirements
| Operation | Target Latency | Max Latency |
|-----------|----------------|-------------|
| Single option pricing | <10ms | <50ms |
| Greeks calculation | <20ms | <100ms |
| Portfolio pricing (100 options) | <500ms | <2s |
| Monte Carlo (10K paths) | <1s | <5s |

### API Design

#### Core Functions
```python
def black_scholes_price(
    S: float,           # Spot price
    K: float,           # Strike price
    T: float,           # Time to expiration (years)
    r: float,           # Risk-free rate
    sigma: float,       # Volatility
    option_type: str    # 'call' or 'put'
) -> float:
    """
    Calculate Black-Scholes option price.

    NO OpenAI models used.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Annualized risk-free rate
        sigma: Annualized volatility
        option_type: 'call' or 'put'

    Returns:
        Option price

    Raises:
        ValueError: If inputs are invalid
    """
    pass

def calculate_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    """
    Calculate all Greeks for an option.

    Returns:
        {
            'delta': float,
            'gamma': float,
            'theta': float,
            'vega': float,
            'rho': float
        }
    """
    pass
```

### Streamlit UI Specification

#### Page Layout
```
Title: <Feature Name>
Subtitle: [Brief description]

Sidebar:
├── Ticker selection (text input)
├── Strike price (number input)
├── Expiration date (date picker)
├── Risk-free rate (slider, 0-10%)
└── Volatility (slider, 0-100%)

Main Panel:
├── Option Price (metric)
├── Greeks Table
│   ├── Delta
│   ├── Gamma
│   ├── Theta
│   ├── Vega
│   └── Rho
├── Price Surface (3D plot)
└── Greeks vs Spot (line charts)
```

## Implementation Phases

### Phase 1: Core Pricing (4 hours)
**Deliverables**:
- `options/<feature>/pricing.py` with Black-Scholes implementation
- Unit tests for pricing accuracy
- Docstrings and type hints

**Validation**:
```bash
pytest tests/test_<feature>_pricing.py -v
```

### Phase 2: Greeks Calculations (3 hours)
**Deliverables**:
- `options/<feature>/greeks.py` with all Greeks
- Unit tests for numerical accuracy
- Performance benchmarks

**Validation**:
```bash
pytest tests/test_<feature>_greeks.py -v
pytest tests/test_<feature>_greeks.py --benchmark-only
```

### Phase 3: Data Integration (2 hours)
**Deliverables**:
- yfinance integration for real-time pricing
- Error handling and retries
- Data caching for performance

**Validation**:
```bash
pytest tests/test_<feature>_data.py -v
python examples/fetch_market_data.py
```

### Phase 4: Streamlit UI (3 hours)
**Deliverables**:
- `streamlit_app/pages/<feature>.py`
- Interactive charts and visualizations
- User input validation

**Validation**:
```bash
streamlit run streamlit_app/pages/<feature>.py
# Manual testing of all inputs
```

### Phase 5: Testing & Documentation (2 hours)
**Deliverables**:
- Complete test suite (>80% coverage)
- README.md updates
- Example Jupyter notebook

**Validation**:
```bash
pytest tests/ --cov=options/<feature> --cov-report=html
open htmlcov/index.html
```

### Phase 6: Integration & Release (1 hour)
**Deliverables**:
- Merge to main branch
- Version tag (e.g., v1.2.0)
- CHANGELOG.md update

**Validation**:
```bash
/validate
git tag v<X.Y.Z>
```

## Quality Gates

### Unit Tests
**Coverage Target**: >80%
**Test Cases**:
- Pricing accuracy vs known values
- Greeks accuracy vs finite differences
- Put-call parity validation
- Boundary conditions (S→0, S→∞, T→0)
- Invalid inputs (negative prices, volatility)

### Integration Tests
- yfinance data fetching
- End-to-end pricing workflow
- Streamlit UI functionality

### Performance Tests
```python
import pytest
from options.<feature> import black_scholes_price

@pytest.mark.benchmark
def test_pricing_performance(benchmark):
    result = benchmark(
        black_scholes_price,
        S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'
    )
    assert result > 0
```

### Security Checks
```bash
# NO OpenAI usage
grep -r "openai\|gpt-" options/ tests/ && exit 1

# API keys only in .env
grep -r "API_KEY\|SECRET" options/ tests/ --include="*.py" && exit 1
```

## Risk Assessment

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical instability | Medium | High | Use log-space calculations, validate inputs |
| yfinance API changes | Low | Medium | Pin version, add integration tests |
| Performance bottleneck | Low | Medium | Optimize NumPy, add caching |
| Division by zero | Medium | Low | Input validation, epsilon guards |

### Business Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Inaccurate pricing | Low | High | Validate vs textbook examples, cross-check |
| Poor adoption | Medium | Medium | User testing, tutorial videos |
| Data costs | Low | Low | Use free yfinance tier |

## Dependencies

### External APIs
- yfinance (free tier): Market data
- No paid APIs required

### Python Libraries
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
streamlit>=1.25.0
yfinance>=0.2.0
scikit-learn>=1.3.0  # Optional, for ML features only
```

**NO OpenAI libraries allowed**

## Testing Strategy

### Test Pyramid
```
      /\
     /UI\         Streamlit manual tests (5%)
    /────\
   /Integ.\       yfinance + end-to-end (15%)
  /────────\
 /   Unit   \     Pricing + Greeks accuracy (80%)
/────────────\
```

### Known Test Values (Black-Scholes)
```python
# Example from Hull's textbook
S=42, K=40, T=0.5, r=0.10, sigma=0.20
Expected call price: 4.76
Expected put price: 0.81
```

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov
      - name: Check for OpenAI
        run: |
          grep -r "openai" . && exit 1 || exit 0
```

## Documentation Requirements

### Code Documentation
- Docstrings for all public functions (Google style)
- Type hints for all function signatures
- Inline comments for complex math

### User Documentation
- README.md: Feature overview and quickstart
- examples/<feature>.ipynb: Jupyter notebook tutorial
- Mathematical appendix: Formula derivations

### Developer Documentation
- PLANNING.md: Architecture decisions
- TASK.md: Current implementation status
- CHANGELOG.md: Version history

## Rollback Plan

### If Tests Fail
```bash
git stash
git checkout main
# Review PRP, fix issues, restart phase
```

### If Performance Issues
- Profile with cProfile
- Optimize NumPy operations
- Add caching layer
- Consider Numba/Cython

### If User Feedback Negative
- Collect specific feedback
- Prioritize fixes
- Release patch version

## Approval

### Sign-off Checklist
- [ ] All tests pass (pytest tests/ -v)
- [ ] NO OpenAI dependencies detected
- [ ] Code coverage >80%
- [ ] Documentation complete
- [ ] Streamlit UI functional
- [ ] Performance targets met
- [ ] Security audit passed

### Reviewers
- Technical Lead: _____________
- Domain Expert: _____________
- QA Engineer: _____________

### Approval Date
Date: ___________
Signature: ___________

---

**Remember**: NO OpenAI models allowed in this project. API keys only in .env file.
