import unittest
import sqlite3
import pandas as pd
import logging
from PriceData.store_trades import df_to_sql_merge_tables_on_date_and_ticker_if_exist


class TestDfToSqlMergeTablesOnDateAndTickerIfExist(unittest.TestCase):
    def setUp(self):
        # Create a mock SQLite database in memory
        self.con = sqlite3.connect(":memory:")
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)
        self.df_new = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "buy_date": ["2023-01-01", "2023-01-02"],
                "current_price": [150.0, 250.0],
                "buy_price": [145.0, 245.0],
                "qty": [10, 20],
                "ratio": [0.5, 0.6],
                "current_vix": [20.0, 21.0],
                "sp500": [4000.0, 4100.0],
                "sell_date": [None, None],
            }
        )

    def test_table_does_not_exist(self):
        strategy_name = "test_strategy"
        df_to_sql_merge_tables_on_date_and_ticker_if_exist(
            self.df_new, strategy_name, self.con, self.logger
        )
        df_result = pd.read_sql(f"SELECT * FROM {strategy_name}", self.con)
        pd.testing.assert_frame_equal(df_result, self.df_new)

    def test_table_exists_and_merge(self):
        strategy_name = "test_strategy"
        df_existing = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "buy_date": ["2023-01-01", "2023-01-03"],
                "current_price": [148.0, 2800.0],
                "buy_price": [145.0, 2750.0],
                "qty": [10, 5],
                "ratio": [0.5, 0.7],
                "current_vix": [19.0, 18.0],
                "sp500": [3990.0, 4200.0],
                "sell_date": [None, None],
            }
        )
        df_existing.to_sql(strategy_name, self.con, index=False)
        df_to_sql_merge_tables_on_date_and_ticker_if_exist(
            self.df_new, strategy_name, self.con, self.logger
        )
        df_result = pd.read_sql(f"SELECT * FROM {strategy_name}", self.con)
        df_expected = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "buy_date": ["2023-01-01", "2023-01-03", "2023-01-02"],
                "current_price": [150.0, 2800.0, 250.0],
                "buy_price": [145.0, 2750.0, 245.0],
                "qty": [10, 5, 20],
                "ratio": [0.5, 0.7, 0.6],
                "current_vix": [20.0, 18.0, 21.0],
                "sp500": [4000.0, 4200.0, 4100.0],
                "sell_date": [None, None, None],
            }
        )
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_handle_exception(self):
        strategy_name = "test_strategy"
        with self.assertLogs(self.logger, level="ERROR") as log:
            df_to_sql_merge_tables_on_date_and_ticker_if_exist(
                None, strategy_name, self.con, self.logger
            )
            self.assertIn("Error while merging data for test_strategy", log.output[0])


if __name__ == "__main__":
    unittest.main()
