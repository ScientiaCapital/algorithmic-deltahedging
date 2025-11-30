# Validate Command - Multi-Phase Validation

**CRITICAL RULE**: NO OpenAI models allowed in this project. Use appropriate alternatives only.

## Purpose
Run comprehensive validation across all quality gates before deployment.

## Usage
```bash
/validate
```

## Validation Phases

### Phase 1: Environment Check
```bash
# Verify Python environment
python --version  # Should be 3.8+
pip list | grep -E "numpy|scipy|pandas|streamlit|scikit-learn|yfinance"

# Check for .env file (API keys only in .env)
test -f .env && echo "✓ .env exists" || echo "⚠ .env missing"

# Verify NO OpenAI dependencies
pip list | grep -i openai && echo "❌ FAIL: OpenAI detected!" || echo "✓ PASS: No OpenAI"
```

### Phase 2: Unit Tests
```bash
# Run full test suite
pytest tests/ -v --tb=short

# Check test coverage
pytest tests/ --cov=options --cov-report=term-missing
```

### Phase 3: Code Quality
```bash
# Type checking (if mypy configured)
python -m mypy options/ --ignore-missing-imports

# Linting
python -m flake8 options/ --max-line-length=100

# Security audit
pip-audit || echo "⚠ pip-audit not installed"
```

### Phase 4: Options Pricing Validation
```bash
# Validate Black-Scholes calculations
pytest tests/test_black_scholes.py -v

# Validate Greeks (delta, gamma, theta, vega)
pytest tests/test_greeks.py -v

# Validate hedging strategies
pytest tests/test_delta_hedging.py -v
```

### Phase 5: Data Pipeline
```bash
# Test yfinance connectivity
python -c "import yfinance as yf; spy = yf.Ticker('SPY'); print('✓ yfinance working')"

# Validate data processing
pytest tests/test_data_pipeline.py -v
```

### Phase 6: Streamlit App
```bash
# Test Streamlit app loads
streamlit run streamlit_app/app.py --server.headless true &
sleep 5
curl -f http://localhost:8501 && echo "✓ Streamlit running" || echo "❌ Streamlit failed"
pkill -f streamlit
```

## Success Criteria
- ✅ All unit tests pass
- ✅ No OpenAI dependencies detected
- ✅ API keys only in .env file
- ✅ Options pricing calculations accurate
- ✅ Greeks calculations validated
- ✅ Streamlit dashboard loads

## Common Issues
1. **Missing dependencies**: `pip install -r requirements.txt`
2. **Test failures**: Check numpy/scipy versions (compatibility)
3. **yfinance errors**: Network connectivity or API rate limits
4. **Streamlit port conflict**: Kill existing process `pkill -f streamlit`

## Next Steps After Validation
- If all phases pass: Run `/generate-prp` for new features
- If failures detected: Fix issues and re-run `/validate`
