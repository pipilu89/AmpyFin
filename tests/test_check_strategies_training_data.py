import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradeSim.testing_random_forest import (
    check_strategies_have_historical_trades,
)


class DummyStrategyA:
    __name__ = "StrategyA"


class DummyStrategyB:
    __name__ = "StrategyB"


class TestCheckStrategiesHaveHistoricalTrades(unittest.TestCase):
    @patch("TradeSim.testing_random_forest.sqlite3.connect")
    @patch("TradeSim.testing_random_forest.pd.read_sql")
    def test_removes_missing_strategies(self, mock_read_sql, mock_connect):
        # Setup
        mock_logger = MagicMock()
        # Only StrategyA exists in the DB
        mock_read_sql.return_value = pd.DataFrame({"name": ["StrategyA"]})
        strategies = [DummyStrategyA, DummyStrategyB]
        trades_list_db_name = "fake_db.sqlite"

        # Act
        result = check_strategies_have_historical_trades(
            strategies, trades_list_db_name, mock_logger
        )

        # Assert
        self.assertEqual(result, [DummyStrategyA])
        mock_logger.warning.assert_called_with(
            "Strategy StrategyB not found in fake_db.sqlite. Removing from strategy array."
        )
        mock_logger.info.assert_any_call(
            "Tables in fake_db.sqlite: ['StrategyA']"
        )
        self.assertIn(
            "len(strategies)", mock_logger.info.call_args_list[-1][0][0]
        )


if __name__ == "__main__":
    unittest.main()
