from math import nan
import unittest
import sqlite3
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PriceData.store_strategy_decisions import df_to_sql_merge_tables_on_date_if_exist


class TestStoreStrategyDecisions(unittest.TestCase):

    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.logger = logging.getLogger("test_logger")

    def tearDown(self):
        self.con.close()

    def test_create_table_if_not_exist(self):
        df_new = pd.DataFrame(
            {"Date": ["2023-01-01", "2023-01-02"], "Ticker1": [1, 2], "Ticker2": [3, 4]}
        ).set_index("Date")

        df_to_sql_merge_tables_on_date_if_exist(
            df_new, "test_strategy", self.con, self.logger
        )

        result = pd.read_sql("SELECT * FROM test_strategy", self.con, index_col="Date")
        pd.testing.assert_frame_equal(result, df_new)

    def test_merge_with_existing_table(self):
        df_existing = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Ticker1": ["buy", "hold"],
                "Ticker2": ["buy", "sell"],
            }
        ).set_index("Date")
        df_existing.to_sql(
            "test_strategy",
            self.con,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )

        df_new = pd.DataFrame(
            {
                "Date": ["2023-01-02", "2023-01-03"],
                "Ticker1": ["hold", "hold"],
                "Ticker2": ["sell", "sell"],
                "Ticker3": ["sell", "sell"],
            }
        ).set_index("Date")

        df_to_sql_merge_tables_on_date_if_exist(
            df_new, "test_strategy", self.con, self.logger
        )

        expected_result = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Ticker3": [None, "sell", "sell"],
                "Ticker1": ["buy", "hold", "hold"],
                "Ticker2": ["buy", "sell", "sell"],
            }
        ).set_index("Date")

        result = pd.read_sql("SELECT * FROM test_strategy", self.con, index_col="Date")
        print(f"{result = }")
        pd.testing.assert_frame_equal(result, expected_result)

    def test_merge_with_nan_values(self):
        df_existing = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Ticker1": ["1", "2"],
                "Ticker2": ["3", "4"],
            }
        ).set_index("Date")
        df_existing.to_sql(
            "test_strategy",
            self.con,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )

        df_new = pd.DataFrame(
            {
                "Date": ["2023-01-02", "2023-01-03"],
                "Ticker1": [None, "6"],
                "Ticker2": ["7", None],
            }
        ).set_index("Date")

        df_to_sql_merge_tables_on_date_if_exist(
            df_new, "test_strategy", self.con, self.logger
        )

        expected_result = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Ticker1": ["1", "2", "6"],
                "Ticker2": ["3", "7", None],
            }
        ).set_index("Date")

        result = pd.read_sql("SELECT * FROM test_strategy", self.con, index_col="Date")
        pd.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
