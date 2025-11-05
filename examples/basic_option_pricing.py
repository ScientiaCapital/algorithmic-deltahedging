"""
Basic Option Pricing Example

This example demonstrates how to price European call and put options
using the Black-Scholes model and view all Greeks.
"""

import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut


def main():
    # Set up option parameters
    current_price = 100.0
    strike_price = 105.0
    volatility = 0.30  # 30% annualized volatility
    risk_free_rate = 0.05  # 5% annual risk-free rate
    drift = 0.10  # 10% expected return
    days_to_expiration = 30

    # Calculate expiration date
    expiration_date = datetime.date.today() + timedelta(days=days_to_expiration)

    print("="*70)
    print("EUROPEAN OPTIONS PRICING EXAMPLE")
    print("="*70)
    print(f"\nMarket Parameters:")
    print(f"  Current Stock Price:    ${current_price:.2f}")
    print(f"  Strike Price:           ${strike_price:.2f}")
    print(f"  Volatility:             {volatility*100:.1f}%")
    print(f"  Risk-Free Rate:         {risk_free_rate*100:.1f}%")
    print(f"  Expected Return (Drift): {drift*100:.1f}%")
    print(f"  Days to Expiration:     {days_to_expiration}")
    print(f"  Expiration Date:        {expiration_date}")

    # Create call option
    print("\n" + "-"*70)
    print("CALL OPTION")
    print("-"*70)

    call_option = EuropeanCall(
        asset_price=current_price,
        strike_price=strike_price,
        volatility=volatility,
        expiration_date=expiration_date,
        risk_free_rate=risk_free_rate,
        drift=drift
    )

    print(f"\nOption Price:  ${call_option.price:.4f}")
    print(f"\nGreeks:")
    print(f"  Delta:  {call_option.delta:>8.4f}  (Sensitivity to $1 price change)")
    print(f"  Gamma:  {call_option.gamma:>8.4f}  (Rate of delta change)")
    print(f"  Vega:   {call_option.vega:>8.4f}  (Sensitivity to 1% vol change)")
    print(f"  Theta:  {call_option.theta:>8.4f}  (Daily time decay)")
    print(f"  Rho:    {call_option.rho:>8.4f}  (Sensitivity to 1% rate change)")
    print(f"\nExercise Probability: {call_option.exercise_prob()*100:.2f}%")

    # Create put option
    print("\n" + "-"*70)
    print("PUT OPTION")
    print("-"*70)

    put_option = EuropeanPut(
        asset_price=current_price,
        strike_price=strike_price,
        volatility=volatility,
        expiration_date=expiration_date,
        risk_free_rate=risk_free_rate,
        drift=drift
    )

    print(f"\nOption Price:  ${put_option.price:.4f}")
    print(f"\nGreeks:")
    print(f"  Delta:  {put_option.delta:>8.4f}  (Sensitivity to $1 price change)")
    print(f"  Gamma:  {put_option.gamma:>8.4f}  (Rate of delta change)")
    print(f"  Vega:   {put_option.vega:>8.4f}  (Sensitivity to 1% vol change)")
    print(f"  Theta:  {put_option.theta:>8.4f}  (Daily time decay)")
    print(f"  Rho:    {put_option.rho:>8.4f}  (Sensitivity to 1% rate change)")
    print(f"\nExercise Probability: {put_option.exercise_prob()*100:.2f}%")

    # Demonstrate put-call parity
    print("\n" + "-"*70)
    print("PUT-CALL PARITY CHECK")
    print("-"*70)

    import math
    parity_lhs = call_option.price - put_option.price
    parity_rhs = current_price - strike_price * math.exp(-risk_free_rate * call_option.dt)

    print(f"\nC - P = {parity_lhs:.4f}")
    print(f"S - K*e^(-rT) = {parity_rhs:.4f}")
    print(f"Difference: {abs(parity_lhs - parity_rhs):.6f}")
    print("(Should be approximately zero)")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
