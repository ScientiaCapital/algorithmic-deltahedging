# Execute PRP Command - 6-Phase Implementation

**CRITICAL RULE**: NO OpenAI models allowed. Use appropriate alternatives only.

## Purpose
Execute a Project Requirements Plan in controlled phases with validation gates.

## Usage
```bash
/execute-prp PRPs/prp_<feature-name>_<YYYYMMDD>.md
```

## Prerequisites
1. PRP file exists and is reviewed
2. `/validate` passes all checks
3. Working branch created: `git checkout -b feat/<feature-name>`

## Phase 1: Setup & Planning
**Duration**: 15 minutes

### Tasks
```bash
# Create feature branch
git checkout -b feat/<feature-name>

# Create test file
touch tests/test_<feature>.py

# Create module file
touch options/<feature>.py

# Update PLANNING.md
echo "## $(date +%Y-%m-%d): Starting <feature-name>" >> PLANNING.md
```

### Validation
- ✅ Branch created
- ✅ Test file exists
- ✅ Module file exists
- ✅ PLANNING.md updated

---

## Phase 2: Core Implementation
**Duration**: 2-4 hours

### Tasks
1. Implement core options pricing logic
2. Add Greeks calculations (delta, gamma, theta, vega)
3. Implement numerical methods (NumPy/SciPy)
4. Add docstrings and type hints

### Code Standards
```python
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple

def black_scholes_price(
    S: float,      # Current stock price
    K: float,      # Strike price
    T: float,      # Time to expiration (years)
    r: float,      # Risk-free rate
    sigma: float,  # Volatility
    option_type: str = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.

    NO OpenAI models used in this calculation.
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
```

### Validation
```bash
# Run tests (should fail initially - TDD)
pytest tests/test_<feature>.py -v

# Type check
python -m mypy options/<feature>.py
```

---

## Phase 3: Testing
**Duration**: 1-2 hours

### Tasks
```python
# tests/test_<feature>.py
import pytest
import numpy as np
from options.<feature> import black_scholes_price

def test_call_option_pricing():
    """Test call option pricing accuracy."""
    # Known values from finance textbooks
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    price = black_scholes_price(S, K, T, r, sigma, 'call')

    # Expected: ~10.45 (theoretical value)
    assert abs(price - 10.45) < 0.01, "Call price accuracy"

def test_put_call_parity():
    """Verify put-call parity holds."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    call = black_scholes_price(S, K, T, r, sigma, 'call')
    put = black_scholes_price(S, K, T, r, sigma, 'put')

    # C - P = S - K*e^(-rT)
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r*T)
    assert abs(parity_lhs - parity_rhs) < 0.01

def test_delta_calculation():
    """Test delta (Greek) accuracy."""
    # Implementation here
    pass
```

### Validation
```bash
pytest tests/test_<feature>.py -v --tb=short
pytest tests/ --cov=options/<feature> --cov-report=term-missing
```

---

## Phase 4: Integration
**Duration**: 1 hour

### Tasks
1. Integrate with existing portfolio module
2. Add to Streamlit dashboard
3. Connect to yfinance data pipeline
4. Update examples/

### Streamlit Integration
```python
# streamlit_app/pages/<feature>.py
import streamlit as st
import yfinance as yf
from options.<feature> import black_scholes_price

st.title("📊 <Feature Name>")

# User inputs
ticker = st.text_input("Ticker", "SPY")
strike = st.number_input("Strike Price", value=100.0)
# ... more inputs

if st.button("Calculate"):
    data = yf.Ticker(ticker)
    current_price = data.history(period="1d")['Close'].iloc[-1]

    price = black_scholes_price(current_price, strike, ...)
    st.metric("Option Price", f"${price:.2f}")
```

### Validation
```bash
# Test Streamlit app
streamlit run streamlit_app/app.py
# Manual testing of new feature
```

---

## Phase 5: Documentation
**Duration**: 30 minutes

### Tasks
1. Update README.md with feature description
2. Add example notebook in `examples/<feature>.ipynb`
3. Update API documentation
4. Add PLANNING.md entry

### Documentation Template
```markdown
## <Feature Name>

### Overview
[Description of what this feature does]

### Usage
\`\`\`python
from options.<feature> import black_scholes_price

price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Option price: ${price:.2f}")
\`\`\`

### Mathematical Background
[Brief explanation of Black-Scholes formula]

### References
- Black, F. and Scholes, M. (1973)
- Hull, J. C. (2018) Options, Futures, and Other Derivatives
```

### Validation
- ✅ README.md updated
- ✅ Example added
- ✅ Docstrings complete

---

## Phase 6: Final Validation & Merge
**Duration**: 30 minutes

### Pre-Merge Checklist
```bash
# 1. Run full test suite
pytest tests/ -v

# 2. Check for OpenAI usage (MUST BE ZERO)
grep -r "openai\|gpt-\|chatgpt" options/ tests/ && echo "❌ FAIL" || echo "✅ PASS"

# 3. Verify .env only contains API keys
grep -v "^#\|^$" .env | grep -i "key\|token\|secret" && echo "✅ Keys in .env"

# 4. Update requirements.txt if needed
pip freeze | grep -E "numpy|scipy|pandas|streamlit" > requirements.txt.new

# 5. Run validation
/validate

# 6. Commit and create PR
git add .
git commit -m "feat(<feature>): Add <description>

- Implement Black-Scholes pricing
- Add Greeks calculations
- Create Streamlit UI
- Add comprehensive tests

NO OpenAI models used."

git push origin feat/<feature-name>
```

### Success Criteria
- ✅ All tests pass
- ✅ Code coverage >80%
- ✅ NO OpenAI dependencies
- ✅ Documentation complete
- ✅ Streamlit UI functional
- ✅ Examples provided

### Merge
```bash
# Create PR
gh pr create --title "feat: <feature-name>" --body "$(cat PRPs/prp_<feature>_*.md)"

# After review, merge
git checkout main
git merge feat/<feature-name>
git tag v<X.Y.Z>
git push origin main --tags
```

## Rollback Plan
If Phase X fails:
```bash
# Stash changes
git stash

# Return to main
git checkout main

# Review PRP and restart from Phase X
```

## Update TASK.md
After each phase, update current status in TASK.md
