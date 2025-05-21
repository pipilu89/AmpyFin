from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import time

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=1)  # clientId must be unique per app
if ib.isConnected():
    print("Connection Successful!")
else:
    print("Connection Failed.")


# contract = Stock("AAPL", "SMART", "USD")
# ticker = ib.reqMktData(contract)
# print(ticker.last)


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


def order_update(trade):
    print(
        f"Order Updated: {trade.contract.localSymbol}, Status: {trade.orderStatus.status}"
    )


if __name__ == "__main__":
    # Ensure asyncio event loop starts
    util.startLoop()

    # Request delayed market data
    ib.reqMarketDataType(3)  # 3 = Delayed data

    # Define AMD stock contract
    contract = Stock("AMD", "SMART", "USD")

    # Request market data
    ticker = ib.reqMktData(contract)
    ib.sleep(5)  # Wait for data to load

    print(f"AMD Last Price: {ticker.last}")
    print(f"Bid: {ticker.bid}, Ask: {ticker.ask}")

    # Request historical data (daily bars)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="30 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
    )

    # Print historical prices
    for bar in bars:
        print(f"Date: {bar.date}, Close Price: {bar.close}")
        # ib.run()
