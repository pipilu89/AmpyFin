import logging
import os
import sys
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test
from TradeSim.utils import precompute_strategy_decisions, load_json_to_dict
from PriceData.store_price_data import sql_to_df_with_date_range
from store_price_data import (
    create_table_schema_strategy_decisions,
    convert_df_to_sql_values,
    upsert_strategy_decisons,
)

from control import (
    test_period_end,
    train_period_start,
    train_tickers,
    regime_tickers,
)

if __name__ == "__main__":
    logger = setup_logging("logs", "store_data.log", level=logging.INFO)

    # create ticker price history from db.
    # table_name = "AAPL"
    # begin_date = "2021-01-04"
    # end_date = "2021-01-29"
    tickers_list = train_tickers + regime_tickers

    ticker_price_history = {}
    for ticker in tickers_list:
        ticker_price_history[ticker] = sql_to_df_with_date_range(
            ticker, train_period_start, test_period_end
        )
    logger.info(f"ticker_price_history: {ticker_price_history}")

    # load/save local copy of ideal_period
    ideal_period_dir = "results"
    ideal_period_filename = f"ideal_period.json"
    ideal_period, _ = load_json_to_dict(ideal_period_dir, ideal_period_filename)
    # logger.info(f"ideal_period: {ideal_period}")

    strategies = strategies_test

    df_precomputed_decisions = precompute_strategy_decisions(
        strategies,
        ticker_price_history,
        train_tickers,
        ideal_period,
        train_period_start,
        test_period_end,
        logger,
    )

    logger.info(f"precomputed_decisions: {df_precomputed_decisions}")

    # last_dl_date_table_name = "last_dl_date"
    # db_path = r"c:\Users\pi\code\python-t212\price_data.db"

    # Ensure the PriceData directory exists
    price_data_dir = "PriceData"
    os.makedirs(price_data_dir, exist_ok=True)

    # Database connection
    strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
    con_sd = sqlite3.connect(strategy_decisions_db_name)

    try:
        for strategy in strategies:
            strategy_name = strategy.__name__
            df_single_strategy = df_precomputed_decisions[
                ["Ticker", "Date", "Action"]
            ].loc[df_precomputed_decisions["Strategy"] == strategy_name]
            logger.info(f"{strategy_name} {df_single_strategy = }")
            # create_table_schema_strategy_decisions(strategy_name, con_sd)
            table_name = create_table_schema_strategy_decisions(strategy_name, con_sd)
            sql_values = convert_df_to_sql_values(
                df_single_strategy, index_boolean=False
            )
            logger.info(f"{sql_values = }")
            upsert_strategy_decisons(strategy_name, sql_values, con_sd)

    except sqlite3.Error as e:
        logger.error(f"database error: {e}")
