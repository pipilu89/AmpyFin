from datetime import datetime
import logging
import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import time
import talib as ta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import (
    setup_logging,
    strategies_test2,
    strategies_test,
    strategies,
    volume_indicators,
)
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
        resolved_columns = {}
        for column in df_new.columns:
            if column in df_existing.columns:
                # Combine the columns and store in a dictionary
                resolved_columns[column] = df_merged[f"{column}_right"].combine_first(
                    df_merged[f"{column}_left"]
                )
                # Drop the original conflicting columns
                df_merged.drop(
                    columns=[f"{column}_left", f"{column}_right"], inplace=True
                )

        # Create a new DataFrame with the resolved columns
        resolved_df = pd.DataFrame(resolved_columns, index=df_merged.index)

        # Concatenate the resolved columns back into the DataFrame
        df_merged = pd.concat([df_merged, resolved_df], axis=1)

        # Resolve conflicts for all columns
        # for column in df_new.columns:
        #     if column in df_existing.columns:
        #         df_merged[column] = df_merged[f"{column}_right"].combine_first(
        #             df_merged[f"{column}_left"]
        #         )
        #         df_merged.drop(
        #             columns=[f"{column}_left", f"{column}_right"],
        #             inplace=True,
        #         )

        # Save merged DataFrame to SQLite database
        df_merged.to_sql(
            strategy_name,
            con,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )
        logger.info(f"Data for {strategy_name} merged and saved to database.")


def convert_decisions_from_by_ticker_to_by_strategy(
    con_source, con_target, strategies, logger
):
    """
    Convert ticker-based strategy decisions to strategy-based tables.

    Parameters:
    con_source (sqlite3.Connection): Source database connection with ticker-based tables
    con_target (sqlite3.Connection): Target database connection for strategy-based tables
    strategies (list): List of strategy functions
    logger (logging.Logger): Logger instance
    """

    logger.info(
        f"==Converting ticker-based strategy decisions to strategy-based tables"
    )
    # Get list of all tickers (tables) from source database
    ticker_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tickers = pd.read_sql(ticker_query, con_source)["name"].tolist()
    logger.info(f"{len(strategies) = } {len(tickers) = }")

    # Initialize dictionary to store strategy DataFrames
    strategy_dfs = {}

    # Process each ticker
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")

        # Read ticker data
        df = pd.read_sql(f"SELECT * FROM '{ticker}'", con_source, index_col="Date")

        # Process each strategy
        for strategy in strategies:
            strategy_name = strategy.__name__

            # Find columns that belong to this strategy
            strategy_cols = [
                col for col in df.columns if col.startswith(f"{strategy_name}")
            ]

            if strategy_cols:
                for col in strategy_cols:
                    # Initialize strategy DataFrame if it doesn't exist
                    if col not in strategy_dfs:
                        strategy_dfs[col] = pd.DataFrame(index=df.index)

                    # Add ticker's data as a new column
                    strategy_dfs[col][ticker] = df[col]

    # Save each strategy DataFrame to the target database
    for strategy_col, df in strategy_dfs.items():
        logger.info(f"Saving strategy table: {strategy_col}")
        df.to_sql(
            strategy_col,
            con_target,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )


def main():
    start_time = time.time()

    tickers_list = train_tickers
    # tickers_list = ["AAPL"]  # test
    # train_tickers = ["AAPL"]  # test

    # Create database connections
    con_pd = sqlite3.connect(PRICE_DB_PATH)

    # Create directories if they don't exist
    price_data_dir = "PriceData"
    os.makedirs(price_data_dir, exist_ok=True)

    # Create intermediate database for ticker-based results
    intermediate_db = os.path.join(price_data_dir, "strategy_decisions_intermediate.db")
    strategy_decisions_db = os.path.join(price_data_dir, "strategy_decisions_final.db")

    con_intermediate = sqlite3.connect(intermediate_db)
    con_final = sqlite3.connect(strategy_decisions_db)

    start_date = datetime.strptime(train_period_start, "%Y-%m-%d")
    start_date_np = np.datetime64(start_date).astype("datetime64[D]")
    train_period_start_with_offset = np.busday_offset(
        start_date_np, -(365 * 2), roll="backward"
    )

    # First pass: compute and store strategy decisions by ticker
    for idx, ticker in enumerate(tickers_list):
        logger.info(
            f"\n=== COMPUTING DECISIONS FOR: {ticker} ({idx + 1}/{len(tickers_list)}) ==="
        )

        strategy_results = []
        # ticker_price_history = sql_to_df_with_date_range(
        #     ticker, train_period_start_with_offset, test_period_end, con_pd
        # )
        ticker_price_history = pd.read_sql_query(
            "SELECT * FROM `{tab}`".format(tab=ticker),
            con_pd,
            index_col="Date",
        )

        for idx, strategy in enumerate(strategies):
            strategy_name = strategy.__name__
            logger.info(f"{strategy_name} ({idx + 1}/{len(strategies)})")
            strategy_result = strategy(ticker_price_history.copy())
            strategy_results.append(strategy_result)

        combined_strategy_results = pd.concat(strategy_results, axis=1)
        df_merged = pd.merge(
            ticker_price_history.copy(),
            combined_strategy_results,
            left_index=True,
            right_index=True,
        )

        desired_order = ["Open", "High", "Low", "Close", "Volume"]
        remaining_columns = [
            col for col in df_merged.columns if col not in desired_order
        ]
        df_merged = df_merged[desired_order + remaining_columns]

        df_to_sql_merge_tables_on_date_if_exist(
            df_merged, ticker, con_intermediate, logger
        )

    # Second pass: convert to strategy-based tables
    logger.info("\n=== Converting to strategy-based tables ===")
    convert_decisions_from_by_ticker_to_by_strategy(
        con_intermediate, con_final, strategies, logger
    )

    # Clean up
    con_pd.close()
    con_intermediate.close()
    con_final.close()

    # Optional: remove intermediate database
    # os.remove(intermediate_db)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    try:
        logger = setup_logging("logs", "store_data.log", level=logging.INFO)
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
