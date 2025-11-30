# algorithmic-deltahedging

**Branch**: master | **Updated**: 2025-11-30

## Status
Feature-complete production-ready Python library for quantitative finance. Comprehensive platform with 8,000+ lines covering options pricing, delta hedging, portfolio management, and risk analytics.

## Today's Focus
1. [ ] Run full test suite and ensure >80% coverage
2. [ ] Create production .env file
3. [ ] Security audit of all dependencies

## Done (This Session)
- (none yet)

## Critical Rules
- **NO OpenAI models** - Use DeepSeek, Qwen, Moonshot via OpenRouter
- API keys in `.env` only, never hardcoded
- Type hints on all functions
- Comprehensive docstrings required
- No hardcoded values (use class parameters)

## Blockers
(none)

## Quick Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=options --cov-report=html

# Run Streamlit dashboard
streamlit run streamlit_app/dashboard.py
```

## Tech Stack
- **Core**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Web Dashboard**: Streamlit
- **ML**: scikit-learn, arch (GARCH), TensorFlow/Keras (optional)
- **Market Data**: yfinance
- **Testing**: pytest, pytest-cov
- **Performance**: numba (JIT compilation)
