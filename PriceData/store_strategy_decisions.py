import logging
import os
import sys
import logging
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test2, strategies_test
from TradeSim.utils import (
    precompute_strategy_decisions,
    load_json_to_dict,
)
from PriceData.store_price_data import sql_to_df_with_date_range

from config import PRICE_DB_PATH

from control import (
    test_period_end,
    train_period_start,
    train_tickers,
    regime_tickers,
)


def df_to_sql_merge_tables_on_date_if_exist(df_new, strategy_name, con, logger):
    """
    Merge a new DataFrame with an existing table in an SQLite database on the 'Date' index.

    If the table does not exist, it creates the table and inserts the new DataFrame.
    If the table exists, it merges the new DataFrame with the existing data on the 'Date' index.
    In case of conflicts, values from the new DataFrame are used unless they are NaN, in which case values from the existing data are kept.

    Parameters:
    df_new (pd.DataFrame): The new DataFrame to be merged with the existing table.
    strategy_name (str): The name of the table in the SQLite database.
    con (sqlite3.Connection): The SQLite database connection object.

    Returns:
    None
    """
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
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )
        logger.info(f"Table {strategy_name} created in the database.")
    else:
        # Load existing data from the database
        existing_data_query = f"SELECT * FROM {strategy_name}"
        df_existing = pd.read_sql(existing_data_query, con, index_col="Date")

        # Merge new data with existing data on index (Date)
        df_merged = pd.merge(
            df_existing,
            df_new,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_left", "_right"),
        )

        # Resolve conflicts for all columns
        for column in df_new.columns:
            if column in df_existing.columns:
                df_merged[column] = df_merged[f"{column}_right"].combine_first(
                    df_merged[f"{column}_left"]
                )
                df_merged.drop(
                    columns=[f"{column}_left", f"{column}_right"],
                    inplace=True,
                )

        # Save merged DataFrame to SQLite database
        df_merged.to_sql(
            strategy_name,
            con,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )
        logger.info(f"Data for {strategy_name} merged and saved to database.")


def main():
    tickers_list = train_tickers + regime_tickers

    # create ticker price history from price db.
    con_pd = sqlite3.connect(PRICE_DB_PATH)
    ticker_price_history = {}
    for ticker in tickers_list:
        ticker_price_history[ticker] = sql_to_df_with_date_range(
            ticker, train_period_start, test_period_end, con_pd
        )
    # logger.info(f"ticker_price_history: {ticker_price_history}")

    # load/save local copy of ideal_period
    ideal_period_dir = "results"
    ideal_period_filename = f"ideal_period.json"
    try:
        ideal_period, _ = load_json_to_dict(ideal_period_dir, ideal_period_filename)
    except Exception as e:
        logger.error(f"Error loading {ideal_period_filename} {e}")

    if not ideal_period:
        return

    # Ensure the PriceData directory exists
    price_data_dir = "PriceData"
    os.makedirs(price_data_dir, exist_ok=True)

    # Database connection
    strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
    con_sd = sqlite3.connect(strategy_decisions_db_name)

    # strategies = strategies_test2
    strategies = strategies_test

    # to reduce problem of large file sizes and memory, we calculate and store each strategy decisions one by one.
    # This way can better handle many tickers and dates. Also can update specific strategy decisions.
    for idx, strategy in enumerate(strategies):
        strategy_name = strategy.__name__
        logger.info(
            f"\n\n=== COMPUTING DECISIONS FOR: {strategy_name} ({idx + 1}/{len(strategies)}) ===\n"
        )
        df_precomputed_decisions = precompute_strategy_decisions(
            strategies,
            ticker_price_history,
            train_tickers,
            ideal_period,
            train_period_start,
            test_period_end,
            logger,
        )

        # format for single strategy
        df_single_strategy = df_precomputed_decisions[["Ticker", "Date", "Action"]].loc[
            df_precomputed_decisions["Strategy"] == strategy_name
        ]

        # pivot df to fit sql table format
        df_pivoted = df_single_strategy.pivot(
            index="Date", columns="Ticker", values="Action"
        )
        # print(df_pivoted)

        df_to_sql_merge_tables_on_date_if_exist(
            df_pivoted, strategy_name, con_sd, logger
        )
    # summary
    logger.info(f"finished")


if __name__ == "__main__":
    try:
        logger = setup_logging("logs", "store_data.log", level=logging.INFO)
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
