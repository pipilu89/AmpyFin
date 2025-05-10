import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import sys
import os
import logging
import logging.config

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from log_config import LOG_CONFIG
from TradeSim.testing_random_forest import (
    strategy_and_ticker_cash_allocation,
    initialize_test_account,
)

# Get the current filename without extension
module_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"log/{module_name}.log"
LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)
# python -m unittest discover -s c:\Users\pi\code\AmpyFin\tests -p "test_strategy_and_ticker_cash_allocation.py"


class TestStrategyAndTickerCashAllocation(unittest.TestCase):
    def setUp(self):
        # Set up the test data
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                "probability": [1, 1, 1, 1],
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )
        self.account = initialize_test_account(10000)
        self.account["holdings"] = {
            # "AAPL": {
            #     "strategy1": {"quantity": 10, "price": 100},
            #     "strategy2": {"quantity": 5, "price": 120},
            # },
            # "MSFT": {
            #     "strategy1": {"quantity": 8, "price": 200},
            #     "strategy2": {"quantity": 3, "price": 220},
            # },
        }
        self.account["cash"] = 10000
        # self.account["total_portfolio_value"] = 15000
        self.account["holdings_value_by_strategy"] = {
            # "strategy1": 3000,
            # "strategy2": 2000,
        }

        self.prediction_threshold = 0.5
        self.asset_limit = 0.25
        self.strategy_limit = 0.5
        # logger = MagicMock()
        # logger = logging.getLogger(__name__)
        self.trade_liquidity_limit_cash = 1000

    # @unittest.skip("Temporarily disabling this test")
    def test_strategy_and_ticker_cash_allocation(self):
        # Call the function with the test data
        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
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
        self.prediction_threshold = 1.1

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
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
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
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
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        # Check that all cash is allocated
        avaliable_cash = self.account["cash"] - self.trade_liquidity_limit_cash
        self.assertEqual(result_df["allocated_cash"].sum(), avaliable_cash)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_with_high_strategy_limit(self):
        # Set a high strategy limit to ensure all cash can be allocated
        self.strategy_limit = 1.0

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        # Check that cash is allocated
        self.assertFalse(result_df.empty)
        for index, row in result_df.iterrows():
            self.assertGreater(row["allocated_cash"], 0)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_capped_by_asset_and_strat_limits(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                # "probability": [0.8, 0.9, 0.85, 0.95],
                "probability": [1, 1, 1, 1],
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )
        self.account = {}
        self.account = {
            "holdings": {
                # "AAPL": {
                #     "strategy1": {"quantity": 10, "price": 150},
                # "strategy2": {"quantity": 5, "price": 120},
                # },
                #     "MSFT": {
                #         "strategy1": {"quantity": 8, "price": 200},
                #         "strategy2": {"quantity": 3, "price": 220},
                # },
            },
            "cash": 10000,
            "total_portfolio_value": 10000,
            # "holdings_value_by_strategy": {"strategy1": 1500},
            "holdings_value_by_strategy": {},
        }

        self.strategy_limit = 0.15
        self.asset_limit = 0.1
        # self.strategy_limit = 0.2
        # self.asset_limit = 0.5
        self.prediction_threshold = 0.5
        self.trade_liquidity_limit_cash = 10

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        print(result_df)

        # Check that cash is allocated
        # check total allocated cash is <= (account cash - liquidity)
        avaliable_cash = self.account["cash"] - self.trade_liquidity_limit_cash
        self.assertLessEqual(result_df["allocated_cash"].sum(), avaliable_cash)

        # check individual strategy allocations are not higher than max_strategy_investment
        max_strategy_investment = (
            self.account["total_portfolio_value"] * self.strategy_limit
        )
        allocation_by_strategy_df = result_df.groupby("strategy_name")[
            "allocated_cash"
        ].sum()
        # print(allocation_by_strategy_df)
        # print(allocation_by_strategy_df.loc["strategy1"])
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy1"], max_strategy_investment
        )
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy2"], max_strategy_investment
        )

        # check individual ticker allocations are below/equal asset_limit_value
        asset_limit_value = (
            self.account["total_portfolio_value"] * self.asset_limit
        )
        allocation_by_ticker_df = result_df.groupby("ticker")[
            "allocated_cash"
        ].sum()
        self.assertLessEqual(
            allocation_by_ticker_df.loc["AAPL"], asset_limit_value
        )
        self.assertLessEqual(
            allocation_by_ticker_df.loc["MSFT"], asset_limit_value
        )

        # limited by both strategy and asset limit
        # strategy1 is fully allocated (equally between 2 tickers) max_strategy_investment = 2250.0,
        # strategy2 allocation is limited by ticker. asset_limit_value = 1500.0
        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.at[0, "allocated_cash"], 750)
        self.assertEqual(result_df.at[1, "allocated_cash"], 750)
        self.assertEqual(result_df.at[2, "allocated_cash"], 250)
        self.assertEqual(result_df.at[3, "allocated_cash"], 250)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_cash_liquidity_limit(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                "probability": [1] * 4,
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )

        self.account = {
            "holdings": {},
            "cash": 10000,
            "total_portfolio_value": 10000,
            "holdings_value_by_strategy": {},
        }

        self.strategy_limit = 0.5
        self.asset_limit = 0.5
        self.prediction_threshold = 0.5
        self.trade_liquidity_limit_cash = 1000

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        print(result_df)
        print(result_df["allocated_cash"].sum())

        # Check that cash is allocated
        # limited by both strategy and asset limit
        # strategy1 is fully allocated (equally between 2 tickers) max_strategy_investment = 2250.0,
        # strategy2 allocation is limited by ticker. asset_limit_value = 1500.0
        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.at[0, "allocated_cash"], 2500)
        self.assertEqual(result_df.at[1, "allocated_cash"], 2500)
        self.assertEqual(result_df.at[2, "allocated_cash"], 2500)
        # capped by cash liquidity limit
        self.assertEqual(result_df.at[3, "allocated_cash"], 1500)
        self.assertEqual(result_df["allocated_cash"].sum(), 9000)

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_with_one_strategy_holding(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                # "probability": [0.8, 0.9, 0.85, 0.95],
                "probability": [1] * 4,
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )

        self.account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {"quantity": 10, "price": 100},
                    # "strategy2": {"quantity": 5, "price": 120},
                },
                #     "MSFT": {
                #         "strategy1": {"quantity": 8, "price": 200},
                #         "strategy2": {"quantity": 3, "price": 220},
                #     },
            },
            "cash": 10000,
            "total_portfolio_value": 11000,
            "holdings_value_by_strategy": {"strategy1": 1000},
            # "holdings_value_by_strategy": {},
        }

        self.strategy_limit = 0.5
        self.asset_limit = 0.5
        self.prediction_threshold = 0.5
        self.trade_liquidity_limit_cash = 1000

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        print(result_df)

        # Check that cash is allocated
        # limited by both strategy and asset limit
        # strategy1 is fully allocated (equally between 2 tickers) max_strategy_investment = 2250.0,
        # strategy2 allocation is limited by ticker. asset_limit_value = 1500.0
        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.at[0, "allocated_cash"], 2250)
        self.assertEqual(result_df.at[1, "allocated_cash"], 2250)
        self.assertEqual(result_df.at[2, "allocated_cash"], 1750)
        self.assertEqual(result_df.at[3, "allocated_cash"], 2750)

        # check total allocated cash is <= (account cash - liquidity)
        avaliable_cash = self.account["cash"] - self.trade_liquidity_limit_cash
        self.assertLessEqual(result_df["allocated_cash"].sum(), avaliable_cash)

        # check individual strategy allocations are not higher than max_strategy_investment
        max_strategy_investment = (
            self.account["total_portfolio_value"] * self.strategy_limit
        )
        allocation_by_strategy_df = result_df.groupby("strategy_name")[
            "allocated_cash"
        ].sum()
        # print(allocation_by_strategy_df)
        # print(allocation_by_strategy_df.loc["strategy1"])
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy1"], max_strategy_investment
        )
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy2"], max_strategy_investment
        )
        # check ticker allocations below/equal asset_limit_value
        asset_limit_value = (
            self.account["total_portfolio_value"] * self.asset_limit
        )
        allocation_by_ticker_df = result_df.groupby("ticker")[
            "allocated_cash"
        ].sum()
        self.assertLessEqual(
            allocation_by_ticker_df.loc["AAPL"], asset_limit_value
        )
        self.assertLessEqual(
            allocation_by_ticker_df.loc["MSFT"], asset_limit_value
        )

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_uses_probabilty(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                # "probability": [0.8, 0.9, 0.85, 0.95],
                "probability": [0.8] * 4,
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )

        self.account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {"quantity": 10, "price": 100},
                    # "strategy2": {"quantity": 5, "price": 120},
                },
                #     "MSFT": {
                #         "strategy1": {"quantity": 8, "price": 200},
                #         "strategy2": {"quantity": 3, "price": 220},
                #     },
            },
            "cash": 10000,
            "total_portfolio_value": 11000,
            "holdings_value_by_strategy": {"strategy1": 1000},
            # "holdings_value_by_strategy": {},
        }

        self.strategy_limit = 0.5
        self.asset_limit = 0.5
        self.prediction_threshold = 0.5
        self.trade_liquidity_limit_cash = 1000

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        print(result_df)

        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.at[0, "allocated_cash"], 1800)
        self.assertEqual(result_df.at[1, "allocated_cash"], 1800)
        self.assertEqual(result_df.at[2, "allocated_cash"], 2200)
        self.assertEqual(result_df.at[3, "allocated_cash"], 2200)

        # check total allocated cash is <= (account cash - liquidity)
        avaliable_cash = self.account["cash"] - self.trade_liquidity_limit_cash
        self.assertLessEqual(result_df["allocated_cash"].sum(), avaliable_cash)

        # check individual strategy allocations are not higher than max_strategy_investment
        max_strategy_investment = (
            self.account["total_portfolio_value"] * self.strategy_limit
        )
        allocation_by_strategy_df = result_df.groupby("strategy_name")[
            "allocated_cash"
        ].sum()
        # print(allocation_by_strategy_df)
        # print(allocation_by_strategy_df.loc["strategy1"])
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy1"], max_strategy_investment
        )
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy2"], max_strategy_investment
        )

        # check individual ticker allocations are below/equal asset_limit_value
        asset_limit_value = (
            self.account["total_portfolio_value"] * self.asset_limit
        )
        allocation_by_ticker_df = result_df.groupby("ticker")[
            "allocated_cash"
        ].sum()
        self.assertLessEqual(
            allocation_by_ticker_df.loc["AAPL"], asset_limit_value
        )
        self.assertLessEqual(
            allocation_by_ticker_df.loc["MSFT"], asset_limit_value
        )

    # @unittest.skip("Temporarily disabling this test")
    def test_allocation_holding_value_higher_than_max_investment(self):
        self.prediction_results_df = pd.DataFrame(
            {
                "strategy_name": [
                    "strategy1",
                    "strategy1",
                    "strategy2",
                    "strategy2",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "action": ["Buy", "Buy", "Buy", "Buy"],
                "prediction": [1, 1, 1, 1],
                # "probability": [0.8, 0.9, 0.85, 0.95],
                "probability": [1] * 4,
                "accuracy": [0.8, 0.9, 0.85, 0.95],
                "current_price": [150, 250, 150, 250],
                "date": ["2025-01-14"] * 4,
            }
        )

        self.account = {
            "holdings": {
                "AAPL": {
                    "strategy1": {"quantity": 9, "price": 1000},  # 9000
                    # "strategy2": {"quantity": 5, "price": 120},
                },
                "MSFT": {
                    # "strategy1": {"quantity": 8, "price": 200},
                    "strategy2": {"quantity": 2, "price": 1000},  # 2000
                },
            },
            "cash": 5000,
            "total_portfolio_value": 16000,
            "holdings_value_by_strategy": {
                "strategy1": 9000,
                "strategy2": 2000,
            },
            # "holdings_value_by_strategy": {},
        }

        self.strategy_limit = 0.5
        self.asset_limit = 0.5
        self.prediction_threshold = 0.5
        self.trade_liquidity_limit_cash = 1000

        result_df = strategy_and_ticker_cash_allocation(
            self.prediction_results_df,
            self.account,
            self.prediction_threshold,
            self.asset_limit,
            self.strategy_limit,
            self.trade_liquidity_limit_cash,
            logger,
        )

        print(result_df[["strategy_name", "allocated_cash", "ticker"]])

        self.assertFalse(result_df.empty)
        # self.assertEqual(result_df.at[0, "allocated_cash"], 1800)
        # self.assertEqual(result_df.at[1, "allocated_cash"], 1800)
        self.assertEqual(result_df.at[2, "allocated_cash"], 3000)
        self.assertEqual(result_df.at[3, "allocated_cash"], 1000)

        # check total allocated cash is <= (account cash - liquidity)
        avaliable_cash = self.account["cash"] - self.trade_liquidity_limit_cash
        self.assertLessEqual(result_df["allocated_cash"].sum(), avaliable_cash)

        # check individual strategy allocations are not higher than max_strategy_investment
        max_strategy_investment = (
            self.account["total_portfolio_value"] * self.strategy_limit
        )
        allocation_by_strategy_df = result_df.groupby("strategy_name")[
            "allocated_cash"
        ].sum()
        # print(allocation_by_strategy_df)
        # print(allocation_by_strategy_df.loc["strategy1"])
        # self.assertLessEqual(
        #     allocation_by_strategy_df.loc["strategy1"], max_strategy_investment
        # )
        self.assertLessEqual(
            allocation_by_strategy_df.loc["strategy2"], max_strategy_investment
        )

        # check individual ticker allocations are below/equal asset_limit_value
        asset_limit_value = (
            self.account["total_portfolio_value"] * self.asset_limit
        )
        allocation_by_ticker_df = result_df.groupby("ticker")[
            "allocated_cash"
        ].sum()
        self.assertLessEqual(
            allocation_by_ticker_df.loc["AAPL"], asset_limit_value
        )
        self.assertLessEqual(
            allocation_by_ticker_df.loc["MSFT"], asset_limit_value
        )


if __name__ == "__main__":
    unittest.main()
