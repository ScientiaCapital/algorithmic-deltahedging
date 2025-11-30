# PLANNING.md - Algorithmic Delta Hedging

**CRITICAL RULE**: NO OpenAI models allowed in this project. Use appropriate alternatives only.

## Project Overview
Quantitative finance library for options pricing, Greeks calculation, and automated delta hedging strategies.

**Tech Stack**:
- Python 3.8+
- NumPy, SciPy, Pandas (scientific computing)
- yfinance (market data)
- Streamlit (visualization)
- scikit-learn (ML features - NO OpenAI)

## Architecture Decisions

### ADR-001: Pricing Models
**Date**: 2025-11-30
**Status**: Adopted

**Context**: Need accurate options pricing for various instruments

**Decision**: Implement multiple pricing models
- Black-Scholes (European options)
- Binomial tree (American options)
- Monte Carlo (exotic options)

**Rationale**:
- Black-Scholes: Industry standard, analytical solution
- Binomial: Handles early exercise
- Monte Carlo: Flexible for complex payoffs

**Consequences**:
- ✅ Comprehensive coverage
- ✅ Educational value
- ⚠️ Increased complexity

### ADR-002: Data Source
**Date**: 2025-11-30
**Status**: Adopted

**Context**: Need reliable, free market data

**Decision**: Use yfinance for market data

**Rationale**:
- Free tier available
- Good API documentation
- Supports stocks, ETFs, options

**Consequences**:
- ✅ Zero data costs
- ⚠️ Rate limits on API
- ⚠️ Dependent on Yahoo Finance uptime

### ADR-003: NO OpenAI Models
**Date**: 2025-11-30
**Status**: **MANDATORY**

**Context**: Project policy prohibits OpenAI usage

**Decision**: NO OpenAI models, libraries, or APIs

**Alternatives**:
- Scientific computing: NumPy, SciPy
- ML (if needed): scikit-learn, local models only
- NLP: spaCy, transformers (local)

**Enforcement**:
```bash
# Pre-commit hook
grep -r "openai\|gpt-" . && exit 1
```

**Consequences**:
- ✅ Cost savings
- ✅ Privacy protection
- ✅ No API dependencies

### ADR-004: Testing Strategy
**Date**: 2025-11-30
**Status**: Adopted

**Decision**: Test-driven development with known values

**Approach**:
- Unit tests: Compare vs textbook examples
- Greeks: Validate vs finite differences
- Performance: Benchmark critical paths

**Known Test Values**:
```python
# Hull's Options textbook (10th ed, Example 13.6)
S=42, K=40, T=0.5, r=0.10, sigma=0.20
Expected call: 4.76
Expected put: 0.81
```

**Consequences**:
- ✅ High confidence in accuracy
- ✅ Regression prevention
- ⚠️ Requires domain knowledge

### ADR-005: API Keys in .env Only
**Date**: 2025-11-30
**Status**: **MANDATORY**

**Context**: Security best practice

**Decision**: All API keys in .env file, never hardcoded

**Enforcement**:
```bash
# Pre-commit hook
grep -r "API_KEY\|SECRET" options/ tests/ --include="*.py" && exit 1
```

**.gitignore**:
```
.env
*.key
credentials.json
```

**Consequences**:
- ✅ Security compliance
- ✅ Easy key rotation
- ⚠️ Requires .env.example documentation

## Module Structure

```
algorithmic-deltahedging/
├── options/                  # Core library
│   ├── __init__.py
│   ├── pricing/
│   │   ├── black_scholes.py # BS pricing model
│   │   ├── binomial.py      # Binomial tree
│   │   └── monte_carlo.py   # MC simulation
│   ├── greeks/
│   │   ├── analytical.py    # Analytical Greeks
│   │   └── numerical.py     # Finite differences
│   ├── hedging/
│   │   ├── delta_hedge.py   # Delta hedging
│   │   └── gamma_scalp.py   # Gamma scalping
│   └── utils/
│       ├── validation.py    # Input validation
│       └── data.py          # yfinance wrappers
├── tests/                    # Test suite
│   ├── test_black_scholes.py
│   ├── test_greeks.py
│   └── test_hedging.py
├── streamlit_app/            # UI
│   ├── app.py
│   └── pages/
│       ├── pricing.py
│       └── hedging.py
├── examples/                 # Tutorials
│   ├── basic_pricing.ipynb
│   └── delta_hedging.ipynb
├── PRPs/                     # Project plans
│   └── templates/
│       └── prp_base.md
├── PLANNING.md              # This file
├── TASK.md                  # Current work
└── requirements.txt
```

## Development Workflow

### Feature Development
1. Create PRP: `/generate-prp <feature>`
2. Validate: `/validate`
3. Execute: `/execute-prp PRPs/prp_<feature>.md`
4. Test: `pytest tests/ -v`
5. Document: Update README, examples
6. Merge: PR with review

### Quality Gates
```bash
# Pre-merge checklist
pytest tests/ -v --cov=options --cov-report=term-missing  # >80% coverage
grep -r "openai" . && exit 1 || exit 0                    # NO OpenAI
python -m mypy options/ --ignore-missing-imports           # Type check
python -m flake8 options/ --max-line-length=100            # Linting
```

## Performance Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Single option pricing | <10ms | <50ms |
| Greeks calculation | <20ms | <100ms |
| Portfolio (100 options) | <500ms | <2s |
| Monte Carlo (10K paths) | <1s | <5s |

## Dependencies

### Required
```
numpy>=1.24.0       # Array operations
scipy>=1.10.0       # Statistical functions
pandas>=2.0.0       # DataFrames
yfinance>=0.2.0     # Market data
streamlit>=1.25.0   # UI
```

### Development
```
pytest>=7.4.0
pytest-cov>=4.1.0
mypy>=1.5.0
flake8>=6.1.0
```

### PROHIBITED
```
openai              # ❌ NEVER
anthropic           # Use Claude Code, not API
langchain-openai    # ❌ NEVER
```

## Mathematical Foundations

### Black-Scholes Formula
```
Call: C = S₀N(d₁) - Ke^(-rT)N(d₂)
Put:  P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
N(x) = cumulative normal distribution
```

### Greeks
```
Delta (Δ):  ∂V/∂S = N(d₁)              [call]
Gamma (Γ):  ∂²V/∂S² = N'(d₁)/(S₀σ√T)
Theta (Θ):  ∂V/∂t = -[S₀N'(d₁)σ/(2√T)] - rKe^(-rT)N(d₂)
Vega (ν):   ∂V/∂σ = S₀N'(d₁)√T
Rho (ρ):    ∂V/∂r = KTe^(-rT)N(d₂)
```

## Timeline & Milestones

### Phase 1: Core Pricing (Completed)
- ✅ Black-Scholes implementation
- ✅ Basic Greeks
- ✅ Unit tests

### Phase 2: Advanced Features (In Progress)
- 🔄 Binomial tree
- 🔄 Monte Carlo
- ⏳ Implied volatility

### Phase 3: Hedging Strategies (Planned)
- ⏳ Delta hedging
- ⏳ Gamma scalping
- ⏳ Portfolio optimization

### Phase 4: Production (Future)
- ⏳ Performance optimization
- ⏳ Documentation complete
- ⏳ PyPI release

## Known Issues

### Numerical Stability
- Division by zero when T → 0
- Overflow in exp() for large values
- **Mitigation**: Input validation, epsilon guards

### Data Quality
- yfinance rate limits
- Missing option data for some tickers
- **Mitigation**: Caching, retry logic

## References

### Textbooks
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.)
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*

### Papers
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach

### Code
- QuantLib (C++ reference implementation)
- Vollib (Python volatility library)

---

**Last Updated**: 2025-11-30
**Next Review**: On feature addition
