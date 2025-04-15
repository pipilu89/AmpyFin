import time

# import pandas as pd
import yfinance as yf
import sqlite3
import os
import sys
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control import train_tickers


def download_and_store(ticker_list, price_data_db_name):
    """Downloads historical OHLCV data for a list of tickers and stores it in a SQLite database.

    Uses yfinance to download the maximum available historical data ('max' period)
    with a '1d' interval. Each ticker's data is stored in a separate table named
    after the ticker symbol within the specified database file. Existing tables
    for the same ticker will be replaced.

    Args:
        ticker_list (list): A list of ticker symbols (strings) to download data for.
        price_data_db_name (str): The file path of the SQLite database where the
                                   price data will be stored.

    Returns:
        tuple: A tuple containing:
            - percentage_of_tickers_saved (float): The percentage of tickers from the
              input list for which data was successfully downloaded and saved.
            - tickers_with_no_data (list): A list of ticker symbols for which
              yfinance did not return any data or an error occurred during saving.
    """
    logger.info(f"start downloading data {len(ticker_list) = }")
    yf_period = "max"
    df = None  # Initialize df
    tickers_saved = []
    tickers_with_no_data = []
    percentage_of_tickers_saved = 0
    try:
        df = yf.download(
            ticker_list,
            group_by="Ticker",
            period=yf_period,
            interval="1d",
            auto_adjust=True,
            repair=True,
            rounding=True,
        )
    except Exception as e:
        logger.error(f"yf error {e}")

    if df is not None:  # Check if df was assigned
        # stack multi-level column index
        df = (
            df.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
        )

        for ticker in ticker_list:
            df_single_ticker = df[["Open", "High", "Low", "Close", "Volume"]].loc[
                df["Ticker"] == ticker
            ]
            df_single_ticker = df_single_ticker.dropna()

            # store ticker in price_data.db
            if df_single_ticker.empty:
                tickers_with_no_data.append(ticker)
                logger.warning(f"no OHLCV data for {ticker}")
            else:
                with sqlite3.connect(price_data_db_name) as conn:
                    try:
                        df_single_ticker.to_sql(ticker, conn, if_exists="replace")
                        tickers_saved.append(ticker)
                    except Exception as e:
                        logger.error(
                            f"error saving {ticker} OHLCV price data to {price_data_db_name}: {e}"
                        )

        percentage_of_tickers_saved = round(
            (len(tickers_saved) / len(ticker_list)) * 100, 2
        )
        logger.info(
            f"{len(tickers_saved)} of {len(ticker_list)} ({percentage_of_tickers_saved} %) tickers saved to {price_data_db_name}"
        )
        if len(tickers_with_no_data) > 0:
            logger.warning(
                f"no data for {len(tickers_with_no_data)} ticker(s): {tickers_with_no_data}"
            )
    return percentage_of_tickers_saved, tickers_with_no_data


def setup_logging(logs_dir, filename, level=logging.INFO):
    """
    Sets up logging to both a file and the console.

    Args:
        logs_dir (str): The directory where the log file will be stored.
        filename (str): The name of the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logging("logs", "store_price_data.log", level=logging.INFO)

    ticker_list = train_tickers
    # # ticker_list = ["APP", "zxz", "AAPL"]
    # ticker_list = ["AAPL"]

    # SQLite database path. Create directories if they don't exist
    """
    Suggest database paths are stored in env/config. Easier to manage different environments.
    """
    price_data_dir = "PriceData"
    os.makedirs(price_data_dir, exist_ok=True)
    PRICE_DB_PATH = os.path.join(price_data_dir, "price_data.db")

    ticker_download_threshold = 90  # %, retry if downloaded tickers pct less than this.
    max_retries = 3
    initial_delay = 30  # seconds
    backoff_factor = 10

    percentage_of_tickers_saved = 0
    tickers_with_no_data = []

    for attempt in range(max_retries):
        percentage_of_tickers_saved, tickers_with_no_data = download_and_store(
            ticker_list, PRICE_DB_PATH
        )

        if percentage_of_tickers_saved >= ticker_download_threshold:
            logger.info(
                f"Ticker download threshold met ({percentage_of_tickers_saved}% >= {ticker_download_threshold}%)"
            )
            break  # Exit loop if successful
        else:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries}: Ticker download threshold not met ({percentage_of_tickers_saved}% < {ticker_download_threshold}%). Retrying..."
            )
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor**attempt)
                logger.info(f"Waiting for {delay:.2f} seconds before next retry.")
                time.sleep(delay)
            else:
                logger.error(
                    f"Max retries ({max_retries}) reached. Ticker download threshold still not met."
                )

    # Final check after all retries
    assert (
        percentage_of_tickers_saved >= ticker_download_threshold
    ), f"Ticker download threshold not met after {max_retries} retries. Only {percentage_of_tickers_saved}% of tickers were saved, which is less than the required {ticker_download_threshold}%. Tickers with no data: {tickers_with_no_data}"
