"""
Portfolio Management Example

This example demonstrates how to manage a multi-position options portfolio
with risk metrics and P&L tracking.
"""

import datetime
from datetime import timedelta
from options.euro_option_analysis import EuropeanCall, EuropeanPut
from options.portfolio import Portfolio, Position, PositionType


def main():
    print("="*70)
    print("OPTIONS PORTFOLIO MANAGEMENT EXAMPLE")
    print("="*70)

    # Create portfolio
    portfolio = Portfolio(name="Sample Options Portfolio")
    portfolio.cash_balance = 100000.0  # Start with $100k cash

    print(f"\nInitial Cash Balance: ${portfolio.cash_balance:,.2f}")

    # Create some options
    expiration = datetime.date.today() + timedelta(days=30)

    # Stock: AAPL
    aapl_call = EuropeanCall(150.0, 155.0, 0.25, expiration, 0.05, 0.10)
    aapl_put = EuropeanPut(150.0, 145.0, 0.25, expiration, 0.05, 0.10)

    # Stock: MSFT
    msft_call = EuropeanCall(380.0, 390.0, 0.28, expiration, 0.05, 0.10)

    print("\n" + "-"*70)
    print("ADDING POSITIONS TO PORTFOLIO")
    print("-"*70)

    # Add long call position on AAPL
    pos1 = Position(
        symbol="AAPL",
        position_type=PositionType.LONG_CALL,
        quantity=5,  # 5 contracts
        entry_price=aapl_call.price,
        entry_date=datetime.date.today(),
        current_price=aapl_call.price,
        option=aapl_call
    )
    portfolio.add_position(pos1)
    print(f"\n1. Long AAPL Call")
    print(f"   Quantity: {pos1.quantity} contracts")
    print(f"   Entry Price: ${pos1.entry_price:.2f}")
    print(f"   Cost: ${pos1.cost_basis:,.2f}")

    # Add long put position on AAPL (protective put)
    pos2 = Position(
        symbol="AAPL",
        position_type=PositionType.LONG_PUT,
        quantity=3,
        entry_price=aapl_put.price,
        entry_date=datetime.date.today(),
        current_price=aapl_put.price,
        option=aapl_put
    )
    portfolio.add_position(pos2)
    print(f"\n2. Long AAPL Put")
    print(f"   Quantity: {pos2.quantity} contracts")
    print(f"   Entry Price: ${pos2.entry_price:.2f}")
    print(f"   Cost: ${pos2.cost_basis:,.2f}")

    # Add short call position on MSFT (covered call)
    pos3 = Position(
        symbol="MSFT",
        position_type=PositionType.SHORT_CALL,
        quantity=-2,  # Short 2 contracts
        entry_price=msft_call.price,
        entry_date=datetime.date.today(),
        current_price=msft_call.price,
        option=msft_call
    )
    portfolio.add_position(pos3)
    print(f"\n3. Short MSFT Call")
    print(f"   Quantity: {abs(pos3.quantity)} contracts (short)")
    print(f"   Entry Price: ${pos3.entry_price:.2f}")
    print(f"   Credit: ${pos3.cost_basis:,.2f}")

    # Add stock position
    pos4 = Position(
        symbol="MSFT",
        position_type=PositionType.LONG_STOCK,
        quantity=200,  # 200 shares
        entry_price=380.0,
        entry_date=datetime.date.today(),
        current_price=380.0
    )
    portfolio.add_position(pos4)
    print(f"\n4. Long MSFT Stock")
    print(f"   Quantity: {pos4.quantity} shares")
    print(f"   Entry Price: ${pos4.entry_price:.2f}")
    print(f"   Cost: ${pos4.cost_basis:,.2f}")

    # Initial portfolio summary
    print("\n" + "-"*70)
    print("INITIAL PORTFOLIO STATUS")
    print("-"*70)
    portfolio.print_summary()

    # Simulate price changes
    print("\n" + "-"*70)
    print("SIMULATING MARKET MOVEMENT")
    print("-"*70)

    print("\nScenario: AAPL +5%, MSFT +3%")

    # Update prices
    new_aapl_price = 150.0 * 1.05  # +5%
    new_msft_price = 380.0 * 1.03  # +3%

    # Recreate options with new prices
    aapl_call_new = EuropeanCall(new_aapl_price, 155.0, 0.25, expiration, 0.05, 0.10)
    aapl_put_new = EuropeanPut(new_aapl_price, 145.0, 0.25, expiration, 0.05, 0.10)
    msft_call_new = EuropeanCall(new_msft_price, 390.0, 0.28, expiration, 0.05, 0.10)

    # Update positions
    pos1.current_price = aapl_call_new.price
    pos1.option = aapl_call_new

    pos2.current_price = aapl_put_new.price
    pos2.option = aapl_put_new

    pos3.current_price = msft_call_new.price
    pos3.option = msft_call_new

    pos4.current_price = new_msft_price

    print(f"\nNew Prices:")
    print(f"  AAPL: ${new_aapl_price:.2f} (+5%)")
    print(f"  MSFT: ${new_msft_price:.2f} (+3%)")

    # Updated portfolio summary
    print("\n" + "-"*70)
    print("UPDATED PORTFOLIO STATUS")
    print("-"*70)
    portfolio.print_summary()

    # Analyze specific positions
    print("\n" + "-"*70)
    print("POSITION-LEVEL ANALYSIS")
    print("-"*70)

    positions_table = portfolio.get_positions_table()
    print(f"\n{'Symbol':<8} {'Type':<15} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L':<12} {'Delta':<10}")
    print("-" * 90)
    for pos_data in positions_table:
        print(f"{pos_data['symbol']:<8} "
              f"{pos_data['type']:<15} "
              f"{pos_data['quantity']:<8.1f} "
              f"${pos_data['entry_price']:<9.2f} "
              f"${pos_data['current_price']:<9.2f} "
              f"${pos_data['unrealized_pnl']:<11.2f} "
              f"{pos_data['delta']:<10.2f}")

    # Close a position
    print("\n" + "-"*70)
    print("CLOSING A POSITION")
    print("-"*70)

    print(f"\nClosing Long AAPL Call position...")
    realized_pnl = portfolio.remove_position(pos1, aapl_call_new.price)
    print(f"Realized P&L: ${realized_pnl:,.2f}")

    # Final summary
    print("\n" + "-"*70)
    print("FINAL PORTFOLIO STATUS")
    print("-"*70)
    portfolio.print_summary()

    print("\n" + "="*70)
    print("\nKey Portfolio Management Features:")
    print("  • Multi-asset position tracking")
    print("  • Aggregate Greeks calculation")
    print("  • Real-time P&L monitoring")
    print("  • Risk metrics (VaR)")
    print("  • Position-level and portfolio-level analysis")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
