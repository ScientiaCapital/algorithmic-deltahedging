# Generate PRP Command

**CRITICAL RULE**: NO OpenAI models allowed. Use appropriate alternatives only.

## Purpose
Create a new Project Requirements Plan (PRP) for feature development.

## Usage
```bash
/generate-prp <feature-name>
```

## Process

### 1. Validate Current State
```bash
# Run validation first
/validate
```

### 2. Gather Context
Ask the user:
- Feature description and objectives
- Target options strategies (e.g., delta hedging, gamma scalping, iron condor)
- Required pricing models (Black-Scholes, Binomial, Monte Carlo)
- Performance requirements (latency, accuracy)
- Data sources (yfinance, custom feeds)
- UI requirements (Streamlit components)

### 3. Create PRP File
```bash
# Generate from template
cp PRPs/templates/prp_base.md PRPs/prp_<feature-name>_<YYYYMMDD>.md
```

### 4. PRP Structure

#### Technical Specification
- **Options Models**: Black-Scholes, Greeks calculations, implied volatility
- **Hedging Strategy**: Delta hedging frequency, rebalancing logic
- **Data Pipeline**: yfinance integration, real-time pricing
- **Calculations**: NumPy/SciPy implementations
- **UI**: Streamlit dashboard components

#### Implementation Phases
1. **Model Development**: Options pricing formulas
2. **Greeks Calculation**: Delta, gamma, theta, vega, rho
3. **Hedging Logic**: Portfolio management, rebalancing
4. **Data Integration**: Market data fetching
5. **Testing**: Unit tests with known option pricing examples
6. **UI**: Streamlit visualization

#### Quality Gates
- Unit tests: `pytest tests/test_<feature>.py -v`
- Pricing accuracy: ±0.01% vs theoretical values
- Greeks accuracy: ±0.001 for delta, ±0.0001 for gamma
- Performance: <100ms for single option pricing
- **NO OpenAI models used**

#### Risk Assessment
- Market data failures (yfinance downtime)
- Numerical stability (division by zero, negative volatility)
- Performance bottlenecks (Monte Carlo simulations)
- API rate limits

### 5. Review with User
Present PRP outline and confirm:
- Scope is clear
- Technical approach is sound
- Success criteria are measurable
- Timeline is realistic

### 6. Save and Tag
```bash
git add PRPs/prp_<feature-name>_<YYYYMMDD>.md
git commit -m "feat: PRP for <feature-name>"
git tag prp-<feature-name>-v1
```

## Template Variables
- `<feature-name>`: Snake_case feature identifier
- `<YYYYMMDD>`: Date stamp
- `<author>`: Developer name
- `<estimated-hours>`: Time estimate

## Example PRPs
- `prp_gamma_scalping_20251130.md` - Gamma scalping strategy
- `prp_implied_volatility_20251130.md` - IV surface calculation
- `prp_monte_carlo_pricing_20251130.md` - MC option pricing

## Next Steps
After PRP approval: Run `/execute-prp <prp-file>`
