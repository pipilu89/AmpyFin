import logging
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test
from TradeSim.utils import (
    prepare_regime_data,
    load_json_to_dict,
    simulate_trading_day,
)
from PriceData.store_price_data import (
    create_table_schema_trades_list,
    convert_df_to_sql_values,
    sql_to_df_with_date_range,
    sql_to_df_with_date_range_no_index,
    upsert_trades_list,
)
from config import PRICE_DB_PATH
from control import (
    test_period_end,
    train_period_start,
    train_period_end,
    train_tickers,
    regime_tickers,
    train_time_delta,
)

if __name__ == "__main__":
    logger = setup_logging("logs", "store_data.log", level=logging.INFO)
    """
    create trades list from strategy decisions
    """
    strategies = strategies_test

    # setup db connections
    price_data_dir = "PriceData"
    strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
    con_sd = sqlite3.connect(strategy_decisions_db_name)

    trades_list_db_name = os.path.join(price_data_dir, "trades_list.db")
    con_tl = sqlite3.connect(trades_list_db_name)

    con_pd = sqlite3.connect(PRICE_DB_PATH)

    # 1. load strategy_decisions from db
    precomputed_decisions = {}
    for strategy in strategies:
        precomputed_decisions[strategy.__name__] = sql_to_df_with_date_range_no_index(
            strategy.__name__, train_period_start, test_period_end, con_sd
        )
        # Hack so works with simulate_trading_day() : add strategy name column
        precomputed_decisions[strategy.__name__]["Strategy"] = strategy.__name__

    logger.info(f"{precomputed_decisions = }")

    # 2. load ticker price history from db.
    tickers_list = train_tickers + regime_tickers
    ticker_price_history = {}
    for ticker in tickers_list:
        ticker_price_history[ticker] = sql_to_df_with_date_range(
            ticker, train_period_start, test_period_end, con_pd
        )
    # === prepare REGIME ma calcs eg 1-day spy return. Use pandas dataframe.
    ticker_price_history = prepare_regime_data(ticker_price_history, logger)

    # 3. run training
    logger.info("Trading simulator and points initialized.")

    start_date = datetime.strptime(train_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(train_period_end, "%Y-%m-%d")
    current_date = start_date

    logger.info(f"Training period: {start_date} to {end_date}")

    trading_simulator = {
        strategy.__name__: {
            "holdings": {},
            "amount_cash": 50000,
            "total_trades": 0,
            "successful_trades": 0,
            "neutral_trades": 0,
            "failed_trades": 0,
            "portfolio_value": 50000,
            "trades_list": [],
        }
        for strategy in strategies
    }

    points = {strategy.__name__: 0 for strategy in strategies}
    time_delta = train_time_delta
    while current_date <= end_date:
        logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")

        if (
            current_date.weekday() >= 5
            or current_date.strftime("%Y-%m-%d")
            not in ticker_price_history[train_tickers[0]].index
        ):
            logger.info(
                f"Skipping {current_date.strftime('%Y-%m-%d')} (weekend or missing data)."
            )
            current_date += timedelta(days=1)
            continue

        trading_simulator, points = simulate_trading_day(
            current_date,
            strategies,
            trading_simulator,
            points,
            time_delta,
            ticker_price_history,
            train_tickers,
            precomputed_decisions,
            logger,
            regime_tickers,
        )
        # Move to next day
        current_date += timedelta(days=1)

    # logger.info(f"{trading_simulator = }")
    # trades_list_all = []

    for strategy in strategies:
        strategy_name = strategy.__name__
        # trades_list_all += trading_simulator[strategy_name]["trades_list"]

        # trades ist df for single strategy
        trades_list_single_strategy_df = pd.DataFrame(
            trading_simulator[strategy_name]["trades_list"],
            columns=[
                "strategy",
                "ticker",
                "current_price",
                "buy_price",
                "qty",
                "ratio",
                "current_vix",
                "sp500",
                "buy_date",
                "sell_date",
            ],
        )
        trades_list_single_strategy_df.drop(columns=["strategy"], inplace=True)

        # put into db
        try:
            trades_list_single_strategy_df.to_sql(
                # strategy_name, con_sd, if_exists="replace", index=False
                strategy_name,
                con_tl,
                if_exists="append",
                index=False,
            )
            # table_name = create_table_schema_trades_list(strategy_name, con_tl)
            # sql_values = convert_df_to_sql_values(
            #     trades_list_single_strategy_df, index_boolean=False
            # )
            # logger.info(f"{sql_values = }")
            # upsert_trades_list(strategy_name, sql_values, con_tl)
        except sqlite3.Error as e:
            logger.error(f"database error: {e}")
