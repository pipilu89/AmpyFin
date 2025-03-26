import sqlite3
import numpy
import talib
import talib as ta
import sys, os
import functools
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup import indicator_periods as ideal_period
from helper_files.client_helper import setup_logging, strategies_test
from PriceData.store_price_data import sql_to_df_with_date_range
from config import PRICE_DB_PATH
from control import (
    benchmark_asset,
    test_period_end,
    test_period_start,
    trade_asset_limit,
    train_start_cash,
    train_stop_loss,
    train_suggestion_heap_limit,
    train_take_profit,
    train_tickers,
    train_time_delta_mode,
    train_trade_liquidity_limit,
    regime_tickers,
    train_trade_asset_limit,
    train_trade_strategy_limit,
    prediction_threshold,
)


def get_historical_data(ticker, current_date, period, ticker_price_history):
    period_start_date = {
        "1mo": current_date - timedelta(days=200),
        # "1mo": current_date - timedelta(days=30),
        "3mo": current_date - timedelta(days=90),
        "6mo": current_date - timedelta(days=180),
        "1y": current_date - timedelta(days=365),
        "2y": current_date - timedelta(days=730),
    }
    start_date = period_start_date[period]

    return ticker_price_history[ticker].loc[: current_date.strftime("%Y-%m-%d")]
    return ticker_price_history[ticker].loc[
        start_date.strftime("%Y-%m-%d") : current_date.strftime("%Y-%m-%d")
    ]


def _process_single_day(
    date, strategies, ticker_price_history, train_tickers, ideal_period
):
    """
    Process a single day for all tickers and strategies.
    This function will be executed in a separate process.
    """
    date_str = date.strftime("%Y-%m-%d")
    result = {
        "date": date_str,
        "strategies": {strategy.__name__: {} for strategy in strategies},
    }

    # Find tickers with data for this date
    available_tickers = [
        ticker
        for ticker in train_tickers
        if date_str in ticker_price_history[ticker].index
    ]

    if not available_tickers:
        return None  # No tickers have data for this date

    # Process each ticker and strategy
    for ticker in available_tickers:
        for strategy in strategies:
            strategy_name = strategy.__name__

            try:
                # Get historical data
                # historical_data = get_historical_data(
                #     ticker, date, ideal_period[strategy_name], ticker_price_history
                # )

                historical_data = ticker_price_history[ticker].loc[:date_str]

                if historical_data is None or historical_data.empty:
                    continue

                # Compute strategy signal
                action = strategy(ticker, historical_data)
                result["strategies"][strategy_name][ticker] = action

            except Exception:
                # Skip errors in worker process
                continue

    return result


def precompute_strategy_decisions(
    strategies,
    ticker_price_history,
    train_tickers,
    ideal_period,
    start_date,
    end_date,
    logger,
):
    """
    Precomputes strategy decisions using parallel processing.
    """
    logger.info("Precomputing strategy decisions with parallel processing...")

    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date -= timedelta(days=1)  # lookahead
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Gather all valid trading days first
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Skip weekends
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    # Prepare parameters for parallel processing
    # We'll process by date to allow better sharing of historical data
    worker_func = functools.partial(
        _process_single_day,
        strategies=strategies,
        ticker_price_history=ticker_price_history,
        train_tickers=train_tickers,
        ideal_period=ideal_period,
    )

    # Use a process pool to parallel process dates
    num_workers = min(cpu_count(), len(trading_days))
    logger.info(f"Using {num_workers} worker processes")

    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, trading_days)

        # Combine results from all processed days into a DataFrame
    data = []
    for day_results in results:
        if day_results:  # Skip empty results
            date_str = day_results["date"]
            for strategy_name, strategy_data in day_results["strategies"].items():
                for ticker, action in strategy_data.items():
                    data.append([strategy_name, ticker, date_str, action])

    df_precomputed_decisions = pd.DataFrame(
        data, columns=["Strategy", "Ticker", "Date", "Action"]
    )

    logger.info(
        f"Strategy decision precomputation complete. Processed {len(results)} trading days."
    )

    return df_precomputed_decisions


def DEMA_indicator(ticker, data):
    """Double Exponential Moving Average (DEMA) indicator."""

    dema = ta.DEMA(data["Close"], timeperiod=30)
    if data["Close"].iloc[-1] > dema.iloc[-1]:
        return "Buy"
    elif data["Close"].iloc[-1] < dema.iloc[-1]:
        return "Sell"
    else:
        return "Hold"


if __name__ == "__main__":
    logger = setup_logging("logs", "indicators_test.log", level=logging.info)
    ticker = "AAPL"
    train_period_start = "2020-01-01"
    train_period_start_window = "2018-01-01"
    test_period_end = "2024-01-01"

    # create ticker price history from price db.
    con_pd = sqlite3.connect(PRICE_DB_PATH)
    data = sql_to_df_with_date_range(
        ticker, train_period_start, test_period_end, con_pd
    )

    print(data)

    # historical_data = get_historical_data(
    #     ticker, current_date, period, ticker_price_history
    # )

    strategies = [strategies_test[2]]
    con_pd = sqlite3.connect(PRICE_DB_PATH)
    ticker_price_history = {}
    for ticker in train_tickers:
        ticker_price_history[ticker] = sql_to_df_with_date_range(
            ticker, train_period_start_window, test_period_end, con_pd
        )

    # Ensure the index of the DataFrame is a datetime object
    ticker_price_history[ticker].index = pd.to_datetime(
        ticker_price_history[ticker].index
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

    logger.info(df_precomputed_decisions)

    # manually calc strategy decisions to see if  the same

    start_date = train_period_start
    end_date = test_period_end
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date -= timedelta(days=1)  # lookahead
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Gather all valid trading days first
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Skip weekends
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    manual_actions = {}
    for date in trading_days:
        # print(f"{date = }")
        # Ensure the index of the DataFrame is a datetime object
        ticker_price_history[ticker].index = pd.to_datetime(
            ticker_price_history[ticker].index
        )
        # print(type(ticker_price_history[ticker].index))  # Should be DatetimeIndex
        # print(type(date))  # Should be datetime.datetime
        # Now slice the DataFrame up to and including the 'date'
        new_start_date = date - timedelta(days=100)
        data = ticker_price_history[ticker].loc[new_start_date:date]
        action = DEMA_indicator(ticker, data)
        # logger.info(f"{action} {date} {len(data) = }")

    # --- Summarization of Actions ---
    logger.info("--- Action Summary ---")

    # 1. Overall Action Counts
    action_counts = df_precomputed_decisions["Action"].value_counts()
    logger.info(f"Overall Action Counts:\n{action_counts}")

    # 2. Action Counts by Strategy
    action_counts_by_strategy = df_precomputed_decisions.groupby(
        ["Strategy", "Action"]
    )["Action"].count()
    logger.info(f"\nAction Counts by Strategy:\n{action_counts_by_strategy}")

    # 3. Action Counts by Ticker
    action_counts_by_ticker = df_precomputed_decisions.groupby(["Ticker", "Action"])[
        "Action"
    ].count()
    logger.info(f"\nAction Counts by Ticker:\n{action_counts_by_ticker}")

    # 4. Action Counts by Date
    action_counts_by_date = df_precomputed_decisions.groupby(["Date", "Action"])[
        "Action"
    ].count()
    logger.info(f"\nAction Counts by Date:\n{action_counts_by_date}")

    # 5. Action Counts by Strategy and Ticker
    action_counts_by_strategy_ticker = df_precomputed_decisions.groupby(
        ["Strategy", "Ticker", "Action"]
    )["Action"].count()
    logger.info(
        f"\nAction Counts by Strategy and Ticker:\n{action_counts_by_strategy_ticker}"
    )
