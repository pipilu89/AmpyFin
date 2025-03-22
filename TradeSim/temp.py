Trades = [
    {
        "symbol": "MSFT",
        "quantity": 12.23,
        "price": 408.73,
        "action": "buy",
        "date": "2024-11-01",
        "strategy": "BOP_indicator",
    },
    {
        "symbol": "AAPL",
        "quantity": 22.48,
        "price": 222.42,
        "action": "buy",
        "date": "2024-11-01",
        "strategy": "BOP_indicator",
    },
]
Holdings = {
    "MSFT": {
        "BOP_indicator": {
            "quantity": 12.23,
            "price": 408.73,
            "stop_loss": 396.47,
            "take_profit": 429.17,
        }
    },
    "AAPL": {
        "BOP_indicator": {
            "quantity": 22.48,
            "price": 222.42,
            "stop_loss": 215.75,
            "take_profit": 233.54,
        }
    },
}

Trades = [
    {
        "symbol": "MSFT",
        "quantity": 12.23,
        "price": 408.73,
        "action": "buy",
        "date": "2024-11-01",
        "strategy": "BOP_indicator",
    },
    {
        "symbol": "AAPL",
        "quantity": 22.48,
        "price": 222.42,
        "action": "buy",
        "date": "2024-11-01",
        "strategy": "BOP_indicator",
    },
    {
        "symbol": "MSFT",
        "quantity": 0.06,
        "price": 406.83,
        "action": "buy",
        "date": "2024-11-04",
        "strategy": "STOCHRSI_indicator",
    },
    {
        "symbol": "AAPL",
        "quantity": 0.09,
        "price": 221.52,
        "action": "buy",
        "date": "2024-11-04",
        "strategy": "BOP_indicator",
    },
]
Holdings = {
    "MSFT": {
        "BOP_indicator": {
            "quantity": 12.23,
            "price": 408.73,
            "stop_loss": 396.47,
            "take_profit": 429.17,
        },
        "STOCHRSI_indicator": {
            "quantity": 0.06,
            "price": 406.83,
            "stop_loss": 394.63,
            "take_profit": 427.17,
        },
    },
    "AAPL": {
        "BOP_indicator": {
            "quantity": 22.57,
            "price": 221.52,
            "stop_loss": 214.87,
            "take_profit": 232.6,
        }
    },
}
