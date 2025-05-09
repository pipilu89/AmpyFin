"""pytest ./tests/test_main_loop.py
what functions should be called?
what if no price/sd data?
test these functions are called correctly
separately check these function work
check trading_account.db has orders (intergration test)
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log_config import LOG_CONFIG
from TradeSim.testing_random_forest import (
    main_test_loop,
    initialize_test_account,
)

# Get the current filename without extension
module_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"log/{module_name}.log"
LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

logging.config.dictConfig(LOG_CONFIG)


def strategy1():
    pass


@pytest.fixture(scope="function")
def test_data():
    logger = logging.getLogger(__name__)
    tickers_list = ["AAPL"]
    use_rf_model_predictions = False
    rf_dict = {}
    experiment_name = "test"
    # One test date
    test_date_range = [datetime(2023, 1, 3)]
    account_values = pd.Series(
        index=pd.date_range(start=test_date_range[0], end=test_date_range[-1])
    )
    trading_account_db_name = os.path.join("tests", "test_trading_account.db")

    # Minimal ticker price history
    ticker_price_history = {
        "AAPL": pd.DataFrame({"Close": [150]}, index=["2023-01-03"]),
        "^VIX": pd.DataFrame({"Close": [20]}, index=["2023-01-03"]),
        "^GSPC": pd.DataFrame(
            {"One_day_spy_return": [0.01]}, index=["2023-01-03"]
        ),
    }
    # Minimal precomputed_decisions
    precomputed_decisions = {
        "strategy1": pd.DataFrame({"AAPL": ["Buy"]}, index=["2023-01-03"])
    }

    trade_liquidity_limit_cash = 1000

    yield logger, tickers_list, use_rf_model_predictions, rf_dict, experiment_name, test_date_range, account_values, trading_account_db_name, ticker_price_history, precomputed_decisions, trade_liquidity_limit_cash


@pytest.fixture(scope="function")
def patched_functions(monkeypatch):
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.generate_tear_sheet",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "TradeSim.testing_random_forest.strategies", [strategy1]
    )

    monkeypatch.setattr(
        "TradeSim.testing_random_forest.create_buy_heap", lambda *a, **k: []
    )

    monkeypatch.setattr(
        "TradeSim.testing_random_forest.insert_account_values_into_db",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.calculate_metrics", lambda *a, **k: {}
    )

    # return first parameter rather than account (the same)
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.process_orders",
        lambda *args: args[1],
    )

    monkeypatch.setattr(
        "TradeSim.testing_random_forest.update_account_portfolio_values",
        lambda a, *args: a,
    )

    return monkeypatch


def test_main_test_loop_basic(monkeypatch, test_data, patched_functions):
    (
        logger,
        tickers_list,
        use_rf_model_predictions,
        rf_dict,
        experiment_name,
        test_date_range,
        account_values,
        trading_account_db_name,
        ticker_price_history,
        precomputed_decisions,
        trade_liquidity_limit_cash,
    ) = test_data

    # Minimal account
    account = initialize_test_account(10000)

    # Patch global variables and dependencies
    mock_check_stop_loss_take_profit_rtn_order = Mock()
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.check_stop_loss_take_profit_rtn_order",
        mock_check_stop_loss_take_profit_rtn_order,
    )

    mock_update_account_portfolio_values = Mock(
        side_effect=lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.update_account_portfolio_values",
        mock_update_account_portfolio_values,
    )

    # execute orders need to return account
    mock_process_orders = MagicMock(
        side_effect=lambda *args, **kwargs: args[1]
    )
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.process_orders",
        mock_process_orders,
    )

    result = main_test_loop(
        account,
        test_date_range,
        tickers_list,
        ticker_price_history,
        precomputed_decisions,
        use_rf_model_predictions,
        account_values,
        trading_account_db_name,
        rf_dict,
        experiment_name,
        trade_liquidity_limit_cash,
        logger,
    )
    assert isinstance(result, dict)
    # check sl_tp, process_orders etc are called and with what parameters.
    assert mock_check_stop_loss_take_profit_rtn_order.call_count == 0
    assert mock_process_orders.call_count == 1
    assert mock_update_account_portfolio_values.call_count == 2


def test_main_test_loop_account_has_holdings(
    monkeypatch, test_data, patched_functions
):
    """If qty of ticker in holding should run sl_tp
    and if action = sell append sell orders"""
    (
        logger,
        tickers_list,
        use_rf_model_predictions,
        rf_dict,
        experiment_name,
        test_date_range,
        account_values,
        trading_account_db_name,
        ticker_price_history,
        precomputed_decisions,
        trade_liquidity_limit_cash,
    ) = test_data

    account = initialize_test_account(10000)
    account["holdings"] = {
        "AAPL": {
            "strategy1": {
                "quantity": 10,
                "price": 150,
            }
        }
    }

    # Patch global variables and dependencies
    mock_check_stop_loss_take_profit_rtn_order = Mock()
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.check_stop_loss_take_profit_rtn_order",
        mock_check_stop_loss_take_profit_rtn_order,
    )

    mock_update_account_portfolio_values = Mock(
        side_effect=lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.update_account_portfolio_values",
        mock_update_account_portfolio_values,
    )

    # execute orders need to return account
    mock_process_orders = MagicMock(
        side_effect=lambda *args, **kwargs: args[1]
    )
    monkeypatch.setattr(
        "TradeSim.testing_random_forest.process_orders",
        mock_process_orders,
    )

    result = main_test_loop(
        account,
        test_date_range,
        tickers_list,
        ticker_price_history,
        precomputed_decisions,
        use_rf_model_predictions,
        account_values,
        trading_account_db_name,
        rf_dict,
        experiment_name,
        trade_liquidity_limit_cash,
        logger,
    )
    assert isinstance(result, dict)
    # check sl_tp, process_orders etc are called and with what parameters.
    assert mock_check_stop_loss_take_profit_rtn_order.call_count == 1
    assert mock_process_orders.call_count == 1
    assert mock_update_account_portfolio_values.call_count == 2
