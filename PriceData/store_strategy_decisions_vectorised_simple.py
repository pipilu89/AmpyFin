import logging
import os
import sys
import logging
import sqlite3
import pandas as pd
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import (
    setup_logging,
    strategies,
)

from config import PRICE_DB_PATH

from control import (
    train_tickers,
)


def main():
    start_time = time.time()

    tickers_list = train_tickers
    # tickers_list = ["AAPL"]  # test

    # Connect to price data db
    # PRICE_DB_PATH = os.path.join("PriceData", "price_data.db")
    con_price_data = sqlite3.connect(PRICE_DB_PATH)

    # Create directories if they don't exist
    price_data_dir = "PriceData"
    os.makedirs(price_data_dir, exist_ok=True)

    # Create strategy_decisions db
    strategy_decisions_db = os.path.join(price_data_dir, "strategy_decisions_test.db")
    con_strategy_decisions = sqlite3.connect(strategy_decisions_db)

    # Compute and store strategy decisions by ticker
    for idx, ticker in enumerate(tickers_list):
        logger.info(
            f"\n=== COMPUTING DECISIONS FOR: {ticker} ({idx + 1}/{len(tickers_list)}) ==="
        )

        strategy_results = []
        ticker_price_history = pd.read_sql_query(
            "SELECT * FROM '{tab}'".format(tab=ticker),
            con_price_data,
            index_col="Date",
        )

        for idx, strategy in enumerate(strategies):
            strategy_name = strategy.__name__
            logger.info(f"{strategy_name} ({idx + 1}/{len(strategies)})")
            strategy_result = strategy(ticker_price_history.copy())
            strategy_results.append(strategy_result)

        combined_strategy_results = pd.concat(strategy_results, axis=1)

        combined_strategy_results.to_sql(
            ticker,
            con_strategy_decisions,
            if_exists="replace",
            index=True,
            dtype={"Date": "DATE PRIMARY KEY"},
        )
        logger.info(f"Data for {ticker} saved to database.")

    # Clean up
    con_price_data.close()
    con_strategy_decisions.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    logger = setup_logging("logs", "store_data.log", level=logging.INFO)
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
