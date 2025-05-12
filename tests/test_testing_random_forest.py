"""
TODO: update execute buy orders
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TradeSim.testing_random_forest import (
    create_buy_heap,
    execute_buy_orders,
    update_account_portfolio_values,
    execute_sell_orders,
    initialize_test_account,
    # test_random_forest,
)


@unittest.skip("Skipping.")
class TestCreateBuyHeap(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "score": [0.9, 0.8, 0.7],
            "allocated_cash": [1000, 800, 600],
            "current_price": [100, 80, 60],
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "strategy_name": ["strategy1", "strategy2", "strategy3"],
            "quantity": [10, 10, 10],
        }
        self.buy_df = pd.DataFrame(data)

    def test_create_buy_heap(self):
        buy_heap = create_buy_heap(self.buy_df)
        # Update the expected_heap according to the new structure
        expected_heap = [
            (-0.9, 10, "AAPL", 100, "strategy1"),
            (-0.8, 10, "MSFT", 80, "strategy2"),
            (-0.7, 10, "GOOGL", 60, "strategy3"),
        ]
        self.assertEqual(buy_heap, expected_heap)

    def test_create_buy_heap_empty(self):
        empty_df = pd.DataFrame(
            columns=[
                "score",
                "allocated_cash",
                "current_price",
                "ticker",
                "strategy_name",
                "quantity",
            ]
        )
        buy_heap = create_buy_heap(empty_df)
        self.assertEqual(buy_heap, [])

        # Add parent directory to sys.path
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )


@unittest.skip("Skipping.")
class TestExecuteBuyOrders(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.train_stop_loss = 0.1
        self.train_take_profit = 0.1
        self.buy_heap = [
            (-0.9, 10, "AAPL", 100, "strategy1"),
            (-0.8, 10, "MSFT", 80, "strategy2"),
            (-0.7, 10, "GOOGL", 60, "strategy3"),
        ]
        self.suggestion_heap = []
        self.account = {
            "holdings": {},
            "cash": 5000,
            "trades": [],
            "total_portfolio_value": 5000,
        }
        self.current_date = datetime.strptime("2023-10-01", "%Y-%m-%d")
        self.train_trade_liquidity_limit = 1000

    def test_execute_buy_orders(self):
        updated_account = execute_buy_orders(
            self.buy_heap,
            self.suggestion_heap,
            self.account,
            self.current_date,
            self.train_trade_liquidity_limit,
            self.train_stop_loss,
            self.train_take_profit,
        )

        print(f"{updated_account =}")

        expected_account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 100,
                        "stop_loss": 90.0,
                        "take_profit": 110.0,
                    }
                },
                "MSFT": {
                    "strategy2": {
                        "quantity": 10,
                        "price": 80,
                        "stop_loss": 72.0,
                        "take_profit": 88.0,
                    }
                },
                "GOOGL": {
                    "strategy3": {
                        "quantity": 10,
                        "price": 60,
                        "stop_loss": 54.0,
                        "take_profit": 66.0,
                    }
                },
            },
            "cash": 5000 - (10 * 100 + 10 * 80 + 10 * 60),
            "trades": [
                {
                    "symbol": "AAPL",
                    "quantity": 10,
                    "price": 100,
                    "action": "buy",
                    "date": "2023-10-01",
                    "strategy": "strategy1",
                },
                {
                    "symbol": "MSFT",
                    "quantity": 10,
                    "price": 80,
                    "action": "buy",
                    "date": "2023-10-01",
                    "strategy": "strategy2",
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 10,
                    "price": 60,
                    "action": "buy",
                    "date": "2023-10-01",
                    "strategy": "strategy3",
                },
            ],
            "total_portfolio_value": 5000,
        }
        # doesn't calc total_portfolio_value. there is a separate function for this.
        self.assertEqual(updated_account, expected_account)

    def test_execute_buy_orders_insufficient_cash(self):
        self.account["cash"] = 100
        updated_account = execute_buy_orders(
            self.buy_heap,
            self.suggestion_heap,
            self.account,
            self.current_date,
            self.train_trade_liquidity_limit,
            self.train_stop_loss,
            self.train_take_profit,
        )
        print(updated_account)
        expected_account = {
            "holdings": {},
            "cash": 100,
            "trades": [],
            "total_portfolio_value": 5000,
        }
        self.assertEqual(updated_account, expected_account)


class TestUpdateAccountPortfolioValues(unittest.TestCase):
    def setUp(self):
        self.starting_cash = 10000
        self.account = initialize_test_account(self.starting_cash)
        self.account["holdings"] = {
            "AAPL": {
                "strategy1": {
                    "quantity": 10,
                    "price": 100,
                }
            },
            "MSFT": {
                "strategy2": {
                    "quantity": 5,
                    "price": 200,
                }
            },
        }
        # self.account["cash"] = 1000
        # self.account["total_portfolio_value"] = 2000

        self.ticker_price_history = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150, 155],
                },
                index=[
                    datetime.strptime("2023-10-01", "%Y-%m-%d"),
                    datetime.strptime("2023-10-02", "%Y-%m-%d"),
                ],
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [250, 255],
                },
                index=[
                    datetime.strptime("2023-10-01", "%Y-%m-%d"),
                    datetime.strptime("2023-10-02", "%Y-%m-%d"),
                ],
            ),
        }
        self.current_date = datetime.strptime("2023-10-01", "%Y-%m-%d")

    def test_update_account_portfolio_values(self):
        updated_account = update_account_portfolio_values(
            self.account, self.ticker_price_history, self.current_date
        )
        expected_account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 100,
                        "current_price": 150.0,
                        "current_value": 1500.0,
                    }
                },
                "MSFT": {
                    "strategy2": {
                        "quantity": 5,
                        "price": 200,
                        "current_price": 250.0,
                        "current_value": 1250.0,
                    }
                },
            },
            "holdings_value_by_strategy": {
                "strategy1": 1500.0,
                "strategy2": 1250.0,
            },
            "cash": 10000,
            "total_portfolio_value": 12750.0,
        }
        print(updated_account)
        print(expected_account)
        self.assertEqual(updated_account, expected_account)

    # @unittest.skip("Skipping. ")
    def test_update_account_portfolio_values_empty_holdings(self):
        self.account = initialize_test_account(self.starting_cash)
        updated_account = update_account_portfolio_values(
            self.account, self.ticker_price_history, self.current_date
        )
        expected_account = {
            "holdings": {},
            "holdings_value_by_strategy": {},
            "cash": self.starting_cash,
            "total_portfolio_value": self.starting_cash,
        }
        self.assertEqual(updated_account, expected_account)

    @unittest.skip("Skipping. how to handle this case?")
    def test_update_account_portfolio_values_no_price_data(self):
        self.current_date = datetime.strptime("2023-10-03", "%Y-%m-%d")
        updated_account = update_account_portfolio_values(
            self.account, self.ticker_price_history, self.current_date
        )
        expected_account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 100,  # 150
                    }
                },
                "MSFT": {
                    "strategy2": {
                        "quantity": 5,
                        "price": 200,  # 250
                    }
                },
            },
            "cash": 1000,
            "total_portfolio_value": 3750,
        }
        self.assertEqual(updated_account, expected_account)


@unittest.skip("Skipping.")
class TestExecuteSellOrders(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.current_date = datetime.strptime("2025-03-22", "%Y-%m-%d")

    def test_execute_sell_orders_success(self):
        account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 150,
                    }
                }
            },
            "cash": 1000,
            "trades": [],
        }
        ticker = "AAPL"
        strategy_name = "strategy1"
        current_price = 200

        updated_account = execute_sell_orders(
            "sell",
            ticker,
            strategy_name,
            account,
            current_price,
            self.current_date,
            "",
            self.logger,
        )

        self.assertEqual(updated_account["cash"], 3000)
        self.assertEqual(len(updated_account["trades"]), 1)
        self.assertNotIn(strategy_name, updated_account["holdings"][ticker])

    def test_execute_sell_orders_no_action(self):
        account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 150,
                    }
                }
            },
            "cash": 1000,
            "trades": [],
        }
        ticker = "AAPL"
        strategy_name = "strategy1"
        current_price = 200

        updated_account = execute_sell_orders(
            "hold",
            ticker,
            strategy_name,
            account,
            current_price,
            self.current_date,
            "",
            self.logger,
        )

        self.assertEqual(updated_account["cash"], 1000)
        self.assertEqual(len(updated_account["trades"]), 0)
        self.assertIn(strategy_name, updated_account["holdings"][ticker])

    def test_execute_sell_orders_ticker_not_in_holdings(self):
        account = {
            "holdings": {
                "MSFT": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 150,
                    }
                }
            },
            "cash": 1000,
            "trades": [],
        }
        ticker = "AAPL"
        strategy_name = "strategy1"
        current_price = 200

        updated_account = execute_sell_orders(
            "sell",
            ticker,
            strategy_name,
            account,
            current_price,
            self.current_date,
            "",
            self.logger,
        )

        self.assertEqual(updated_account["cash"], 1000)
        self.assertEqual(len(updated_account["trades"]), 0)
        self.assertNotIn(ticker, updated_account["holdings"])

    def test_execute_sell_orders_strategy_not_in_holdings(self):
        account = {
            "holdings": {
                "AAPL": {
                    "strategy2": {
                        "quantity": 10,
                        "price": 150,
                    }
                }
            },
            "cash": 1000,
            "trades": [],
        }
        ticker = "AAPL"
        strategy_name = "strategy1"
        current_price = 200

        updated_account = execute_sell_orders(
            "sell",
            ticker,
            strategy_name,
            account,
            current_price,
            self.current_date,
            "",
            self.logger,
        )

        self.assertEqual(updated_account["cash"], 1000)
        self.assertEqual(len(updated_account["trades"]), 0)
        self.assertIn("strategy2", updated_account["holdings"][ticker])


if __name__ == "__main__":
    unittest.main()
