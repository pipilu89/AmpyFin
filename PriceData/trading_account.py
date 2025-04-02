import pandas as pd

# Sample Data
data = {
    "TransactionID": [1, 2, 3, 4],
    "TransactionDate": ["2025-03-01", "2025-03-05", "2025-03-10", "2025-03-15"],
    "TransactionType": ["Buy", "Sell", "Buy", "Dividend"],
    "TickerSymbol": ["AAPL", "AAPL", "TSLA", None],
    "Quantity": [10, -5, 20, None],  # Negative quantity for a sale
    "Price": [150.00, 155.00, 900.00, None],
    "TotalValue": [1500.00, -775.00, 18000.00, 50.00],  # Transaction value
    "AccountID": [101, 101, 101, 101],
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure TransactionDate is a datetime object
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

# Display the DataFrame
print(df)


profit_loss = df["TotalValue"].sum()
print(f"Overall Profit/Loss: {profit_loss}")

returns_per_asset = df.groupby("TickerSymbol")["TotalValue"].sum()
print(f"{returns_per_asset = }")

holdings = df.groupby("TickerSymbol")["Quantity"].sum()
print(f"{holdings = }")


def weighted_avg_cost(symbol):
    asset_data = df[df["TickerSymbol"] == symbol]
    buy_transactions = asset_data[asset_data["TransactionType"] == "Buy"]
    return (
        buy_transactions["Quantity"] * buy_transactions["Price"]
    ).sum() / buy_transactions["Quantity"].sum()


# Example for a single asset:
ticker = "AAPL"
wac = weighted_avg_cost(ticker)
print(f"Weighted Average Cost for {ticker}: {wac}")


sells = df[df["TransactionType"] == "Sell"]
win_rate = (sells["TotalValue"] > 0).mean() * 100
print(f"Win Rate: {win_rate:.2f}%")


# Cumulative Value over time
import matplotlib.pyplot as plt

df["CumulativeValue"] = df["TotalValue"].cumsum()
plt.plot(df["TransactionDate"], df["CumulativeValue"])
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value Over Time")
# plt.show()
