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


def df_to_sql_merge_tables_on_date_and_ticker_if_exist(
    df_new, strategy_name, con, logger
):
    """
    Merge a new DataFrame with an existing table in an SQLite database on the 'ticker' and 'buy_date' columns.

    If the table does not exist, it creates the table and inserts the new DataFrame.
    If the table exists, it merges the new DataFrame with the existing data on the 'ticker' and 'buy_date' columns.
    In case of conflicts, values from the new DataFrame are used unless they are NaN, in which case values from the existing data are kept.

    Parameters:
    df_new (pd.DataFrame): The new DataFrame to be merged with the existing table.
    strategy_name (str): The name of the table in the SQLite database.
    con (sqlite3.Connection): The SQLite database connection object.
    logger (logging.Logger): The logger object for logging information.

    Returns:
    None
    """
    dtype_mapping = {
        "ticker": "TEXT",
        "buy_date": "DATE",
        "current_price": "REAL",
        "buy_price": "REAL",
        "qty": "INTEGER",
        "ratio": "REAL",
        "current_vix": "REAL",
        "sp500": "REAL",
        "sell_date": "DATE",
    }

    # Check if the table exists
    table_exists_query = (
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{strategy_name}'"
    )
    table_exists = pd.read_sql(table_exists_query, con)

    if table_exists.empty:
        # Create the table if it doesn't exist
        df_new.to_sql(
            strategy_name,
            con,
            if_exists="replace",
            index=False,
            dtype=dtype_mapping,
        )
        logger.info(f"Table {strategy_name} created in the database.")

        # Create an index on the primary key columns
        con.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{strategy_name}_ticker_buy_date ON {strategy_name} (ticker, buy_date)"
        )
    else:
        # Load existing data from the database
        existing_data_query = f"SELECT * FROM {strategy_name}"
        df_existing = pd.read_sql(existing_data_query, con)

        # Merge new data with existing data on 'ticker' and 'buy_date'
        df_merged = pd.merge(
            df_existing,
            df_new,
            on=["ticker", "buy_date"],
            how="outer",
            suffixes=("_left", "_right"),
        )

        # Resolve conflicts for all columns except 'ticker' and 'buy_date'
        for column in df_new.columns:
            if column in df_existing.columns and column not in ["ticker", "buy_date"]:
                df_merged[column] = df_merged[f"{column}_right"].combine_first(
                    df_merged[f"{column}_left"]
                )
                df_merged.drop(
                    columns=[f"{column}_left", f"{column}_right"],
                    inplace=True,
                )

        # Drop the suffixes from 'ticker' and 'buy_date' columns
        df_merged = df_merged.rename(
            columns={"ticker_left": "ticker", "buy_date_left": "buy_date"}
        )

        logger.debug(f"{df_merged = }")
        # Save merged DataFrame to SQLite database
        df_merged.to_sql(
            strategy_name, con, if_exists="replace", index=False, dtype=dtype_mapping
        )
        logger.info(f"Data for {strategy_name} merged and saved to database.")


if __name__ == "__main__":
    logger = setup_logging("logs", "trades_list.log", level=logging.WARNING)
    """
    create trades list from strategy decisions
    """
    strategies = strategies_test
    # strategies = [strategies_test[3]]
    tickers_list = train_tickers + regime_tickers
    logger.info(f"=== START COMPUTE TRADES LIST ===")
    logger.info(f"{len(train_tickers) = } {len(regime_tickers) = }\n")

    # setup db connections
    price_data_dir = "PriceData"
    strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
    con_sd = sqlite3.connect(strategy_decisions_db_name)

    trades_list_db_name = os.path.join(price_data_dir, "trades_list.db")
    con_tl = sqlite3.connect(trades_list_db_name)

    con_pd = sqlite3.connect(PRICE_DB_PATH)

    # 2. load ticker price history from db.
    ticker_price_history = {}
    for ticker in tickers_list:
        ticker_price_history[ticker] = sql_to_df_with_date_range(
            ticker, train_period_start, test_period_end, con_pd
        )
    # === prepare REGIME ma calcs eg 1-day spy return. Use pandas dataframe.
    logger.info(f"prepare regime data")
    ticker_price_history = prepare_regime_data(ticker_price_history, logger)

    # 3. run training
    start_date = datetime.strptime(train_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(train_period_end, "%Y-%m-%d")

    logger.info(f"Training period: {start_date} to {end_date}")

    number_of_trades_by_strategy = {}
    for idx, strategy in enumerate(strategies):
        current_date = start_date
        strategy_name = strategy.__name__
        logger.info(
            f"\n\n=== COMPUTING TRADES FOR: {strategy_name} ({idx + 1}/{len(strategies)}) ===\n"
        )
        # logger.info(f"{strategy_name} loop start...")

        # 1. load strategy_decisions from db
        existing_data_query = f"SELECT * FROM {strategy_name}"
        df_existing = pd.read_sql(existing_data_query, con_sd, index_col="Date")

        # Melt df_existing to stack ticker columns to rows
        precomputed_decisions = df_existing.reset_index().melt(
            id_vars=["Date"], var_name="Ticker", value_name="Action"
        )
        logger.debug(f"{precomputed_decisions = }")

        trading_simulator = {
            strategy_name: {
                "holdings": {},
                "amount_cash": 50000,
                "total_trades": 0,
                "successful_trades": 0,
                "neutral_trades": 0,
                "failed_trades": 0,
                "portfolio_value": 50000,
                "trades_list": [],
            }
            # for strategy in strategies
        }

        # points = {strategy_name: 0 for strategy in strategies}
        points = {strategy_name: 0}
        time_delta = train_time_delta
        while current_date <= end_date:
            # logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")

            if (
                current_date.weekday() >= 5
                or current_date.strftime("%Y-%m-%d")
                not in ticker_price_history[train_tickers[0]].index
            ):
                logger.debug(
                    f"Skipping {current_date.strftime('%Y-%m-%d')} (weekend or missing data)."
                )
                current_date += timedelta(days=1)
                continue

            trading_simulator, points = simulate_trading_day(
                current_date,
                # strategies,
                strategy,
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

        logger.info(f"{points = }")
        # logger.info(f"{trading_simulator = }")
        # trades_list_all = []

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

        number_of_trades = len(trades_list_single_strategy_df)
        # put into db
        if number_of_trades > 0:
            logger.info(f"{strategy_name} {number_of_trades = }")

            try:
                df_to_sql_merge_tables_on_date_and_ticker_if_exist(
                    trades_list_single_strategy_df, strategy_name, con_tl, logger
                )
            except Exception as e:
                logger.error(f"update db error {e}")

        else:
            logger.info(f"{strategy_name}, no trades.")

        # summary
        number_of_trades_by_strategy[strategy_name] = {number_of_trades}

    logger.info(f"\n\n=== SUMMARY ===\n")
    logger.info(f"{number_of_trades_by_strategy = }")
