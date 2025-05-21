from ib_insync import IB, Stock, MarketOrder

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=1)  # Change to 4001 for live trading

if ib.isConnected():
    print("Connected to IBKR API successfully!")
else:
    print("Connection failed.")

# Get Account Summary
# account_summary = ib.accountSummary()
# print(account_summary)

# Get Account Values Individually
# print(ib.accountValues())

# Get Portfolio Holdings
# portfolio = ib.portfolio()
# for position in portfolio:
#     print(position)

# Get Cash Balance
cash_balance = next(
    (item.value for item in ib.accountSummary() if item.tag == "CashBalance"),
    None,
)
print("Cash Balance:", cash_balance)

# cash_balance = ib.accountSummary().get("CashBalance")
# print("Cash Balance:", cash_balance)
# accountSummary = ib.accountSummary()
# print("accountSummary:", accountSummary[0])

# Loop through all summary fields:
# Get account summary
# account_summary = ib.accountSummary()

# Iterate through the list
# for item in account_summary:
#     print(f"{item.tag}: {item.value} {item.currency}")


def place_trade_ibkr(ticker, side, qty):
    # To place a trade using ib_insync
    # Define stock contract
    contract = Stock(ticker, "SMART", "USD")

    # Define order (Market Buy Order)
    order = MarketOrder(side, qty)  # Buy 10 shares

    # Place the order
    trade = ib.placeOrder(contract, order)

    print("Trade Submitted:", trade)

    ib.sleep(2)  # Wait for execution
    print("Order Status:", trade.orderStatus.status)
    orderId = trade.orderStatus.orderId
    print(f"{orderId=}")
    # print("OrderId:", trade.orderStatus.orderId)
    return trade


# trade = place_trade_ibkr("AAPL", "BUY", 1)


# Retrieve Open Orders - This will return active (unfilled) orders:
open_orders = ib.openOrders()

if open_orders:
    for order in open_orders:
        print(f"{order=}")
        # print(
        #     f"Order: {order.contract.localSymbol}, {order.order.action} {order.order.totalQuantity}, Status: {order.orderStatus.status}"
        # )
else:
    print("No open orders.")

# Retrieve Open Portfolio Positions. This returns currently held positions:
portfolio = ib.portfolio()

if portfolio:
    for position in portfolio:
        print(
            f"Position: {position.contract.localSymbol}, Quantity: {position.position}, Avg Price: {position.avgCost}"
        )
else:
    print("No open positions.")

# Check Order by ID
order_id = 15  # Replace with actual order ID
# order_id = trade.orderStatus.orderId

# Fetch all orders and filter by ID
order = next((o for o in ib.orders() if o.orderId == order_id), None)

if order:
    print(
        f"Order ID {order_id}: {order.contract.localSymbol}, {order.order.action} {order.order.totalQuantity}, Status: {order.orderStatus.status}"
    )
else:
    print(f"Order ID {order_id} not found.")


# order_status = ib.reqOrderState(order_id)
# print(order_status)


# Event-Driven Order Monitoring (Async Approach) For real-time updates without polling:
def order_update(trade):
    print(
        f"Order Updated: {trade.contract.localSymbol}, Status: {trade.orderStatus.status}"
    )


ib.orderStatusEvent += order_update  # Listen for updates


# Continuously Check for Pending Orders. To fetch unfilled orders and monitor their status dynamically:
import time


def check_pending_orders():
    while True:
        pending_orders = [
            order
            for order in ib.orders()
            if hasattr(order, "orderStatus")
            and order.orderStatus.status in ("PendingSubmit", "Submitted")
        ]

        if pending_orders:
            print("\nPending Orders:")
            for order in pending_orders:
                print(
                    f"Order ID {order.orderId}: {order.action} {order.totalQuantity} {order.contract.localSymbol}, Status: {order.orderStatus.status}"
                )

        else:
            print("No pending orders.")

        time.sleep(5)  # Check every 5 seconds


# Run monitoring loop
check_pending_orders()
