import logging
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test, strategies
from TradeSim.utils import (
    prepare_regime_data,
    simulate_trading_day,
)
from PriceData.store_price_data import (
    sql_to_df_with_date_range,
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
    df_new: pd.DataFrame,
    strategy_name: str,
    con: sqlite3.Connection,
    logger: logging.Logger,
) -> None:
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

    def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
        query = (
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        return not pd.read_sql(query, con).empty

    def create_table(
        df: pd.DataFrame, table_name: str, con: sqlite3.Connection, dtype_mapping: dict
    ) -> None:
        df.to_sql(
            table_name, con, if_exists="replace", index=False, dtype=dtype_mapping
        )
        con.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_ticker_buy_date ON {table_name} (ticker, buy_date)"
        )
        logger.info(f"Table {table_name} created in the database.")

    def merge_dataframes(
        df_existing: pd.DataFrame, df_new: pd.DataFrame
    ) -> pd.DataFrame:
        df_merged = pd.merge(
            df_existing,
            df_new,
            on=["ticker", "buy_date"],
            how="outer",
            suffixes=("_left", "_right"),
        )
        for column in df_new.columns:
            if column in df_existing.columns and column not in ["ticker", "buy_date"]:
                df_merged[column] = df_merged[f"{column}_right"].combine_first(
                    df_merged[f"{column}_left"]
                )
                df_merged.drop(
                    columns=[f"{column}_left", f"{column}_right"], inplace=True
                )
        return df_merged.rename(
            columns={"ticker_left": "ticker", "buy_date_left": "buy_date"}
        )

    try:
        if not table_exists(con, strategy_name):
            create_table(df_new, strategy_name, con, dtype_mapping)
        else:
            df_existing = pd.read_sql(f"SELECT * FROM {strategy_name}", con)
            df_merged = merge_dataframes(df_existing, df_new)
            df_merged.to_sql(
                strategy_name,
                con,
                if_exists="replace",
                index=False,
                dtype=dtype_mapping,
            )
            logger.info(f"Data for {strategy_name} merged and saved to database.")
    except Exception as e:
        logger.error(f"Error while merging data for {strategy_name}: {e}")


if __name__ == "__main__":
    logger = setup_logging("logs", "trades_list.log", level=logging.WARNING)
    """
    create trades list from strategy decisions
    """
    # strategies = strategies_test
    strategies = [strategies_test[3]]
    tickers_list = train_tickers + regime_tickers
    logger.info(f"=== START COMPUTE TRADES LIST ===")
    logger.info(f"{len(train_tickers) = } {len(regime_tickers) = }\n")

    start_date = datetime.strptime(train_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(train_period_end, "%Y-%m-%d")

    # Subtract 3 day3 - needed for sp500 1 day return. 3 days because wkends
    # train_period_start_minus_three_days = start_date - timedelta(days=3)

    # Subtract 1 business day from start_date
    start_date_np = np.datetime64(start_date).astype("datetime64[D]")
    start_date_minus_one_business_day = np.busday_offset(
        start_date_np, -1, roll="backward"
    )
    start_date_minus_one_business_day_str = start_date_minus_one_business_day.astype(
        str
    )

    logger.info(f"Training period: {start_date} to {end_date}")
    logger.info(
        f"Start date minus one business day: {start_date_minus_one_business_day_str}"
    )

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
            ticker, start_date_minus_one_business_day_str, test_period_end, con_pd
        )
    # === prepare REGIME ma calcs eg 1-day spy return. Use pandas dataframe.
    logger.info(f"prepare regime data")
    ticker_price_history = prepare_regime_data(ticker_price_history, logger)
    # logger.info(f"{ticker_price_history = }")

    # 3. run training
    logger.info(f"Training period: {start_date} to {end_date}")

    number_of_trades_by_strategy = {}
    for idx, strategy in enumerate(strategies):
        start_time = time.time()  # Record the start time
        current_date = start_date
        strategy_name = strategy.__name__
        logger.info(
            f"\n\n=== COMPUTING TRADES FOR: {strategy_name} ({idx + 1}/{len(strategies)}) ===\n"
        )
        # logger.info(f"{strategy_name} loop start...")

        # 1. load strategy_decisions from db
        existing_data_query = f"SELECT * FROM {strategy_name}"
        precomputed_decisions = pd.read_sql(
            existing_data_query, con_sd, index_col="Date"
        )

        assert (
            precomputed_decisions.index.name == "Date"
        ), "Index of precomputed_decisions is not set to 'Date'"

        # logger.info(f"{precomputed_decisions = }")

        # Melt df_existing to stack ticker columns to rows
        # precomputed_decisions = df_existing.reset_index().melt(
        #     id_vars=["Date"], var_name="Ticker", value_name="Action"
        # )

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
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")

    logger.info(f"\n\n=== SUMMARY ===\n")
    logger.info(f"{number_of_trades_by_strategy = }")
