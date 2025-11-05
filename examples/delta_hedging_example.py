"""
Delta Hedging Strategy Example

This example demonstrates how to create and manage a delta-neutral
hedge for an option position.
"""

import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall
from options.delta_hedging import DeltaHedgingStrategy


def main():
    print("="*70)
    print("DELTA HEDGING STRATEGY EXAMPLE")
    print("="*70)

    # Create an option to hedge
    expiration = datetime.date.today() + timedelta(days=30)
    call_option = EuropeanCall(
        asset_price=100.0,
        strike_price=105.0,
        volatility=0.30,
        expiration_date=expiration,
        risk_free_rate=0.05,
        drift=0.10
    )

    print(f"\nOption to Hedge:")
    print(f"  Type:          Long Call")
    print(f"  Quantity:      1 contract (100 shares)")
    print(f"  Strike:        ${call_option.strike_price:.2f}")
    print(f"  Current Price: ${call_option.asset_price:.2f}")
    print(f"  Option Price:  ${call_option.price:.4f}")
    print(f"  Delta:         {call_option.delta:.4f}")

    # Create hedging strategy
    strategy = DeltaHedgingStrategy(
        rebalance_threshold=10.0,  # Rebalance if delta exceeds 10 shares
        commission_per_share=0.005,  # $0.005 per share
        slippage_bps=5.0  # 5 basis points slippage
    )

    print(f"\nHedging Strategy Parameters:")
    print(f"  Rebalance Threshold: {strategy.rebalance_threshold} shares")
    print(f"  Commission:          ${strategy.commission_per_share:.3f} per share")
    print(f"  Slippage:            {strategy.slippage_bps} bps")

    # Create initial hedge
    print(f"\n" + "-"*70)
    print("CREATING INITIAL HEDGE")
    print("-"*70)

    position = strategy.create_hedge(
        option=call_option,
        option_quantity=1,  # Long 1 contract
        current_stock_price=100.0,
        target_delta=0.0  # Delta-neutral
    )

    print(f"\nInitial Hedge Established:")
    print(f"  Option Position:     {position.option_quantity} contract")
    print(f"  Stock Position:      {position.stock_quantity:.2f} shares")
    print(f"  Initial Delta:       {position.current_delta:.4f}")
    print(f"  Target Delta:        {position.target_delta:.4f}")
    print(f"  Transaction Cost:    ${position.transactions[0].cost:.2f}")

    # Simulate price movement and rebalancing
    print(f"\n" + "-"*70)
    print("SIMULATING PRICE MOVEMENT AND REBALANCING")
    print("-"*70)

    # Scenario 1: Price moves up to $105
    print(f"\nScenario 1: Stock price moves to $105")
    new_price_1 = 105.0

    # Create new option with updated price
    updated_call_1 = EuropeanCall(
        asset_price=new_price_1,
        strike_price=call_option.strike_price,
        volatility=call_option.volatility,
        expiration_date=expiration,
        risk_free_rate=call_option.risk_free_rate,
        drift=call_option.drift
    )
    position.option = updated_call_1

    print(f"  New Stock Price:     ${new_price_1:.2f}")
    print(f"  New Option Delta:    {updated_call_1.delta:.4f}")
    print(f"  Current Portfolio Delta: {position.current_delta:.4f}")
    print(f"  Delta Imbalance:     {position.delta_imbalance:.4f}")

    # Check if rebalancing is needed
    needs_rebalance = strategy.check_rebalance_needed(position)
    print(f"  Rebalancing Needed:  {needs_rebalance}")

    if needs_rebalance:
        txn = strategy.rebalance_position(position, new_price_1)
        if txn:
            print(f"\nRebalancing Transaction:")
            print(f"  Type:        {txn.transaction_type.value}")
            print(f"  Quantity:    {txn.quantity:.2f} shares")
            print(f"  Price:       ${txn.price:.2f}")
            print(f"  Cost:        ${txn.cost:.2f}")
            print(f"  New Stock Position: {position.stock_quantity:.2f} shares")
            print(f"  New Portfolio Delta: {position.current_delta:.4f}")

    # Calculate P&L
    print(f"\n" + "-"*70)
    print("PROFIT & LOSS ANALYSIS")
    print("-"*70)

    pnl = strategy.calculate_pnl(
        position=position,
        current_stock_price=new_price_1,
        current_option_price=updated_call_1.price
    )

    print(f"\nP&L Breakdown:")
    print(f"  Option P&L:          ${pnl['option_pnl']:>10.2f}")
    print(f"  Stock P&L:           ${pnl['stock_pnl']:>10.2f}")
    print(f"  Transaction Costs:   ${pnl['transaction_costs']:>10.2f}")
    print(f"  " + "-"*40)
    print(f"  Net P&L:             ${pnl['net_pnl']:>10.2f}")
    print(f"  Current Delta:       {pnl['current_delta']:>10.4f}")

    # Strategy summary
    print(f"\n" + "-"*70)
    print("STRATEGY SUMMARY")
    print("-"*70)

    summary = strategy.get_portfolio_summary()
    print(f"\nTotal Positions:     {summary['total_positions']}")
    print(f"Portfolio Delta:     {summary['total_delta']:.4f}")
    print(f"Transaction Costs:   ${summary['total_transaction_costs']:.2f}")

    print("\n" + "="*70)
    print("\nKey Takeaways:")
    print("  1. Delta hedging neutralizes directional risk")
    print("  2. Positions need periodic rebalancing as delta changes")
    print("  3. Transaction costs reduce overall profitability")
    print("  4. The hedge isolates other Greeks (gamma, vega, theta)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
