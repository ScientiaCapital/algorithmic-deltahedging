"""
Options Trading Dashboard - Streamlit Application

A comprehensive web-based dashboard for options pricing, Greeks analysis,
portfolio management, and strategy visualization.

To run: streamlit run streamlit_app/dashboard.py
"""

import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import our options library
import sys
sys.path.append('.')
from options.euro_option_analysis import EuropeanCall, EuropeanPut
from options.american_options import AmericanCall, AmericanPut
from options.implied_volatility import implied_volatility, calculate_volatility_smile
from options.strategies import *
from options.market_data import YFinanceProvider, get_market_data_summary
from options.portfolio import Portfolio, Position, PositionType
from options.risk_metrics import PortfolioRiskAnalyzer


# Page configuration
st.set_page_config(
    page_title="Options Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 Algorithmic Delta Hedging Dashboard</p>',
           unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Options Pricing", "Implied Volatility", "Greeks Analysis",
     "Strategy Builder", "Portfolio Manager", "Risk Analytics", "Market Data"]
)

# ==================== OPTIONS PRICING PAGE ====================
if page == "Options Pricing":
    st.header("Options Pricing Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        option_style = st.selectbox("Option Style", ["European", "American"])
        option_type = st.selectbox("Option Type", ["Call", "Put"])

        asset_price = st.number_input("Current Asset Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=105.0, min_value=0.01)
        volatility = st.slider("Volatility (%)", min_value=1, max_value=100, value=30) / 100
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0, max_value=20, value=5) / 100

        days_to_exp = st.number_input("Days to Expiration", value=30, min_value=1, max_value=365)
        expiration_date = datetime.date.today() + timedelta(days=int(days_to_exp))

        if option_style == "American":
            dividend_yield = st.slider("Dividend Yield (%)", min_value=0, max_value=10, value=0) / 100
        else:
            drift = st.slider("Drift (%)", min_value=-20, max_value=50, value=10) / 100

    with col2:
        st.subheader("Option Price & Greeks")

        try:
            # Create option
            if option_style == "European":
                if option_type == "Call":
                    option = EuropeanCall(asset_price, strike_price, volatility,
                                         expiration_date, risk_free_rate, drift)
                else:
                    option = EuropeanPut(asset_price, strike_price, volatility,
                                        expiration_date, risk_free_rate, drift)
            else:  # American
                if option_type == "Call":
                    option = AmericanCall(asset_price, strike_price, volatility,
                                        expiration_date, risk_free_rate, dividend_yield)
                else:
                    option = AmericanPut(asset_price, strike_price, volatility,
                                       expiration_date, risk_free_rate, dividend_yield)

            # Display results
            st.metric("Option Price", f"${option.price:.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Delta", f"{option.delta:.4f}")
                st.metric("Gamma", f"{option.gamma:.4f}")
                st.metric("Vega", f"{option.vega:.4f}")
            with col_b:
                st.metric("Theta", f"{option.theta:.4f}")
                st.metric("Rho", f"{option.rho:.4f}")

        except Exception as e:
            st.error(f"Error calculating option price: {str(e)}")

    # Payoff diagram
    st.subheader("Payoff Diagram at Expiration")

    price_range = np.linspace(strike_price * 0.5, strike_price * 1.5, 100)
    if option_type == "Call":
        intrinsic = np.maximum(price_range - strike_price, 0) - option.price
    else:
        intrinsic = np.maximum(strike_price - price_range, 0) - option.price

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=intrinsic, mode='lines',
                            name='Profit/Loss', line=dict(width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=strike_price, line_dash="dot", line_color="green")
    fig.update_layout(
        xaxis_title="Asset Price at Expiration ($)",
        yaxis_title="Profit/Loss ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================== IMPLIED VOLATILITY PAGE ====================
elif page == "Implied Volatility":
    st.header("Implied Volatility Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        option_type_iv = st.selectbox("Option Type", ["Call", "Put"], key="iv_type")
        market_price = st.number_input("Market Option Price ($)", value=5.0, min_value=0.01)
        asset_price_iv = st.number_input("Current Asset Price ($)", value=100.0,
                                        min_value=0.01, key="iv_asset")
        strike_price_iv = st.number_input("Strike Price ($)", value=105.0,
                                         min_value=0.01, key="iv_strike")
        days_to_exp_iv = st.number_input("Days to Expiration", value=30,
                                        min_value=1, max_value=365, key="iv_days")
        risk_free_rate_iv = st.slider("Risk-Free Rate (%)", min_value=0,
                                      max_value=20, value=5, key="iv_rf") / 100
        method = st.selectbox("Calculation Method",
                            ["Newton-Raphson", "Bisection", "Brent's Method"])

    with col2:
        st.subheader("Results")

        if st.button("Calculate Implied Volatility"):
            try:
                method_map = {
                    "Newton-Raphson": "newton",
                    "Bisection": "bisection",
                    "Brent's Method": "brent"
                }

                time_to_exp = days_to_exp_iv / 365.0

                iv = implied_volatility(
                    option_price=market_price,
                    asset_price=asset_price_iv,
                    strike_price=strike_price_iv,
                    time_to_expiration=time_to_exp,
                    risk_free_rate=risk_free_rate_iv,
                    option_type=option_type_iv.lower(),
                    method=method_map[method]
                )

                st.success(f"Implied Volatility: {iv*100:.2f}%")

                # Display comparison
                st.info(f"This represents the market's expectation of future volatility "
                       f"for {option_type_iv} option with strike ${strike_price_iv:.2f}")

            except Exception as e:
                st.error(f"Error calculating implied volatility: {str(e)}")


# ==================== GREEKS ANALYSIS PAGE ====================
elif page == "Greeks Analysis":
    st.header("Greeks Sensitivity Analysis")

    option_type_greeks = st.selectbox("Option Type", ["Call", "Put"], key="greeks_type")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Asset Price", value=100.0, key="greeks_S")
        K = st.number_input("Strike Price", value=100.0, key="greeks_K")
    with col2:
        vol = st.slider("Volatility (%)", 1, 100, 30, key="greeks_vol") / 100
        r = st.slider("Risk-Free Rate (%)", 0, 20, 5, key="greeks_r") / 100
    with col3:
        days = st.number_input("Days to Expiration", value=30, key="greeks_days")

    # Create price range for analysis
    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    expiration = datetime.date.today() + timedelta(days=int(days))

    # Calculate Greeks for each price
    deltas, gammas, vegas, thetas = [], [], [], []

    for price in price_range:
        if option_type_greeks == "Call":
            opt = EuropeanCall(price, K, vol, expiration, r, 0.1)
        else:
            opt = EuropeanPut(price, K, vol, expiration, r, 0.1)

        deltas.append(opt.delta)
        gammas.append(opt.gamma)
        vegas.append(opt.vega)
        thetas.append(opt.theta)

    # Plot Greeks
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=price_range, y=deltas, name='Delta', yaxis='y1'))
    fig.add_trace(go.Scatter(x=price_range, y=gammas, name='Gamma', yaxis='y2'))
    fig.add_trace(go.Scatter(x=price_range, y=vegas, name='Vega', yaxis='y3'))
    fig.add_trace(go.Scatter(x=price_range, y=thetas, name='Theta', yaxis='y4'))

    fig.update_layout(
        xaxis=dict(domain=[0.1, 0.9], title="Asset Price"),
        yaxis=dict(title="Delta", side="left"),
        yaxis2=dict(title="Gamma", overlaying="y", side="right"),
        yaxis3=dict(title="Vega", overlaying="y", side="right", position=0.95),
        yaxis4=dict(title="Theta", overlaying="y", side="right", position=1.0),
        title="Greeks Sensitivity Analysis",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# ==================== STRATEGY BUILDER PAGE ====================
elif page == "Strategy Builder":
    st.header("Options Strategy Builder")

    strategy_type = st.selectbox(
        "Select Strategy",
        ["Bull Call Spread", "Bear Put Spread", "Long Straddle",
         "Iron Condor", "Butterfly Spread"]
    )

    st.subheader("Strategy Parameters")

    # Common parameters
    asset_price_strat = st.number_input("Current Asset Price", value=100.0, key="strat_price")
    vol_strat = st.slider("Volatility (%)", 1, 100, 30, key="strat_vol") / 100
    days_strat = st.number_input("Days to Expiration", value=30, key="strat_days")
    expiration_strat = datetime.date.today() + timedelta(days=int(days_strat))
    r_strat = 0.05

    st.info(f"Building {strategy_type} strategy...")

    # This is a simplified example - full implementation would create actual strategies
    st.write("Strategy implementation coming soon!")


# ==================== PORTFOLIO MANAGER PAGE ====================
elif page == "Portfolio Manager":
    st.header("Portfolio Management")

    # Initialize session state for portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio(name="My Portfolio")
        st.session_state.portfolio.cash_balance = 100000.0

    portfolio = st.session_state.portfolio

    # Display portfolio summary
    st.subheader("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cash Balance", f"${portfolio.cash_balance:,.2f}")
    with col2:
        st.metric("Positions", len(portfolio.positions))
    with col3:
        st.metric("Total P&L", f"${portfolio.total_pnl:,.2f}")
    with col4:
        st.metric("Portfolio Delta", f"{portfolio.portfolio_delta:.2f}")

    # Add new position
    with st.expander("Add New Position"):
        pos_symbol = st.text_input("Symbol", value="AAPL")
        pos_type = st.selectbox("Position Type",
                               ["Long Call", "Short Call", "Long Put", "Short Put"])
        pos_quantity = st.number_input("Quantity", value=1, min_value=1)
        pos_price = st.number_input("Entry Price", value=5.0, min_value=0.01)

        if st.button("Add Position"):
            st.success("Position added!")

    # Show positions table
    if portfolio.positions:
        st.subheader("Current Positions")
        positions_data = portfolio.get_positions_table()
        st.dataframe(pd.DataFrame(positions_data))


# ==================== RISK ANALYTICS PAGE ====================
elif page == "Risk Analytics":
    st.header("Risk Analytics & VaR Calculation")

    # Generate sample returns for demonstration
    np.random.seed(42)
    portfolio_values = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 252)))

    analyzer = PortfolioRiskAnalyzer(portfolio_values)
    metrics = analyzer.calculate_all_metrics()

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Portfolio Value", f"${metrics['current_value']:,.2f}")
        st.metric("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")

    with col2:
        st.metric("Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']*100:.2f}%")
        st.metric("VaR (95%)", f"${metrics.get('historical_var_95', 0):,.2f}")

    # VaR comparison chart
    st.subheader("Value at Risk Comparison")
    var_data = {
        'Method': ['Historical', 'Parametric', 'Monte Carlo'],
        '95% VaR': [
            metrics.get('historical_var_95', 0),
            metrics.get('parametric_var_95', 0),
            metrics.get('monte_carlo_var_95', 0)
        ],
        '99% VaR': [
            metrics.get('historical_var_99', 0),
            metrics.get('parametric_var_99', 0),
            metrics.get('monte_carlo_var_99', 0)
        ]
    }

    var_df = pd.DataFrame(var_data)
    fig = px.bar(var_df, x='Method', y=['95% VaR', '99% VaR'], barmode='group',
                title="VaR by Method and Confidence Level")
    st.plotly_chart(fig, use_container_width=True)


# ==================== MARKET DATA PAGE ====================
elif page == "Market Data":
    st.header("Real-Time Market Data")

    symbol = st.text_input("Enter Stock Symbol", value="AAPL")

    if st.button("Fetch Data"):
        try:
            provider = YFinanceProvider()

            with st.spinner("Fetching market data..."):
                summary = get_market_data_summary(symbol, provider)

            st.success(f"Data fetched for {symbol}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Current Price", f"${summary['current_price']:.2f}")
                st.metric("Dividend Yield", f"{summary['dividend_yield']*100:.2f}%")

            with col2:
                st.metric("Historical Vol (30d)", f"{summary['historical_volatility_30d']*100:.2f}%")
                st.metric("Risk-Free Rate", f"{summary['risk_free_rate']*100:.2f}%")

            # Available expirations
            st.subheader("Available Option Expirations")
            expirations = summary.get('available_expirations', [])
            if expirations:
                st.write(", ".join(expirations[:10]))  # Show first 10
            else:
                st.info("No options data available")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Algorithmic Delta Hedging Dashboard | "
    "Built with Streamlit | "
    "© 2025</p>"
    "</div>",
    unsafe_allow_html=True
)
