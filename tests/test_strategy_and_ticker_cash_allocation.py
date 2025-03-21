import unittest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TradeSim.testing_random_forest import strategy_and_ticker_cash_allocation


# python -m unittest discover -s c:\Users\pi\code\AmpyFin\tests -p "test_strategy_and_ticker_cash_allocation.py"


class TestStrategyAndTickerCashAllocation(unittest.TestCase):
    def setUp(self):
        # Set up the test data
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": ["strategy1", "strategy1", "strategy2", "strategy2"],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
            }
        )

        self.account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {"quantity": 10, "price": 100},
                    "strategy2": {"quantity": 5, "price": 120},
                },
                "MSFT": {
                    "strategy1": {"quantity": 8, "price": 200},
                    "strategy2": {"quantity": 3, "price": 220},
                },
            },
            "cash": 10000,
            "trades": [],
            "total_portfolio_value": 15000,
        }

        self.holdings_value_by_strategy = {"strategy1": 3000, "strategy2": 2000}

        self.prediction_threshold = 0.5
        self.asset_limit = 0.25
        self.strategy_limit = 0.5

    # @unittest.skip("Temporarily disabling this test")
    def test_strategy_and_ticker_cash_allocation(self):
        # Call the function with the test data
        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        # Check the results
        self.assertFalse(result_df.empty)
        self.assertIn("strategy_name", result_df.columns)
        self.assertIn("ticker", result_df.columns)
        self.assertIn("allocated_cash", result_df.columns)

        # Add more assertions as needed to verify the correctness of the results
        for index, row in result_df.iterrows():
            self.assertGreater(row["allocated_cash"], 0)

    # @unittest.skip("Temporarily disabling this test")
    def test_no_qualifying_strategies(self):
        # Set a high prediction threshold to ensure no strategies qualify
        self.prediction_threshold = 1.0

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        # Check that the result is empty
        self.assertTrue(result_df.empty)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_with_zero_cash(self):
        # Set account cash to zero
        self.account["cash"] = 0

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        print(result_df)
        # Check that no cash is allocated
        for index, row in result_df.iterrows():
            self.assertEqual(row["allocated_cash"], 0)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_with_high_asset_limit(self):
        # Set a high asset limit to ensure all cash can be allocated
        self.asset_limit = 1.0

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        # Check that cash is allocated
        self.assertFalse(result_df.empty)
        for index, row in result_df.iterrows():
            self.assertGreater(row["allocated_cash"], 0)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_with_high_strategy_limit(self):
        # Set a high strategy limit to ensure all cash can be allocated
        self.strategy_limit = 1.0

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        # Check that cash is allocated
        self.assertFalse(result_df.empty)
        for index, row in result_df.iterrows():
            self.assertGreater(row["allocated_cash"], 0)

    def test_allocation_with_one_strategy_holding(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": ["strategy1", "strategy1", "strategy2", "strategy2"],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
            }
        )

        self.account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {"quantity": 10, "price": 150},
                    # "strategy2": {"quantity": 5, "price": 120},
                },
                #     "MSFT": {
                #         "strategy1": {"quantity": 8, "price": 200},
                #         "strategy2": {"quantity": 3, "price": 220},
                #     },
            },
            "cash": 10000,
            "trades": [],
            "total_portfolio_value": 15000,
        }

        self.holdings_value_by_strategy = {"strategy1": 1000}
        self.strategy_limit = 0.1
        self.asset_limit = 0.15

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.holdings_value_by_strategy,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
        )

        # print(result_df)

        # Check that cash is allocated
        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.at[0, "allocated_cash"], 200)
        self.assertEqual(result_df.at[1, "allocated_cash"], 225)
        # limited by both strategy and asset limit
        self.assertEqual(result_df.at[2, "allocated_cash"], 550)
        self.assertEqual(result_df.at[3, "allocated_cash"], 712.5)
        for index, row in result_df.iterrows():
            self.assertGreater(row["allocated_cash"], 0)


if __name__ == "__main__":
    unittest.main()
