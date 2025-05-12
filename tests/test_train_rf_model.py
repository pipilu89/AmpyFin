import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import logging
import logging.config

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from log_config import LOG_CONFIG
from TradeSim.testing_random_forest import (
    get_trades_training_data_from_db,
    prepare_sp500_one_day_return,
)

# Get the current filename without extension
module_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"log/{module_name}.log"
with open(log_filename, "w"):
    pass
LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


# @unittest.skip("Skipping.")
class Test_get_trades_training_data_from_db(unittest.TestCase):
    def setUp(self):
        self.strategy_name = "ULTOSC_indicator"
        self.trades_list_db_name = os.path.join(
            "PriceData", "trades_list_vectorised.db"
        )

    def test_trade_data_from_db(self):
        start_date = "2020-01-01"
        end_date = "2020-05-01"
        result_df = get_trades_training_data_from_db(
            self.strategy_name,
            self.trades_list_db_name,
            logger,
            start_date,
            end_date,
        )

        # print(result_df)
        self.assertEqual(len(result_df), 2)

    def test_just_end_date(self):
        start_date = None
        end_date = "2020-05-01"
        result_df = get_trades_training_data_from_db(
            self.strategy_name,
            self.trades_list_db_name,
            logger,
            start_date,
            end_date,
        )

        # print(result_df)
        self.assertEqual(len(result_df), 1750)

    def test_invalid_strategyname(self):
        start_date = None
        end_date = "2020-05-01"
        strategy_name = "invalid_name"
        result_df = get_trades_training_data_from_db(
            strategy_name,
            self.trades_list_db_name,
            logger,
            None,
            end_date,
        )

        # print(result_df)
        self.assertEqual(len(result_df), 0)


if __name__ == "__main__":

    unittest.main()
