import numpy as np
import timeit
from numba import njit
import talib as ta
import pandas as pd
import sqlite3
import os, sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRICE_DB_PATH

from store_strategy_decisions_vectorised_simple import main


# --- Numba-Accelerated Version ---
@njit
def generate_signals_numba(condition_buy, condition_sell, default=0):
    n = condition_buy.shape[0]
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        if condition_buy[i]:
            result[i] = 1
        elif condition_sell[i]:
            result[i] = -1
        else:
            result[i] = default
    return result


# --- Helper Function for Common Logic ---
def generate_signals_orig(condition_buy, condition_sell, default="Hold"):
    """Uses np.select to generate signals based on boolean conditions."""
    conditions = [condition_buy, condition_sell]
    choices = ["Buy", "Sell"]
    return np.select(conditions, choices, default=default)


def generate_signals(condition_buy, condition_sell, default=0):
    """Uses Numba-accelerated method to generate signals based on boolean conditions."""
    return generate_signals_numba(condition_buy, condition_sell, default)


# Original numpy version for comparison
def generate_signals_np(condition_buy, condition_sell, default=0):
    conditions = [condition_buy, condition_sell]
    choices = [1, -1]
    return np.select(conditions, choices, default=default)


def test_signal_performance(condition_buy, condition_sell):
    """Test correctness and performance of signal generation methods."""

    # Original numpy version for comparison
    def generate_signals_np(condition_buy, condition_sell, default=0):
        conditions = [condition_buy, condition_sell]
        choices = [1, -1]
        return np.select(conditions, choices, default=default)

    # --- Check for Correctness ---
    signals_np = generate_signals_np(condition_buy, condition_sell)

    # Compile the Numba function (the first call includes compilation overhead)
    signals_numba = generate_signals_numba(condition_buy, condition_sell)

    # Verify that both outputs are identical
    if np.array_equal(signals_np, signals_numba):
        print("Signal arrays are equal: Correct!")
    else:
        print("Signal arrays differ: There is an error.")

    # --- Profile the Performance ---
    # Run the Numba function once more to eliminate compilation time in timing
    _ = generate_signals_numba(condition_buy, condition_sell)

    # Time both implementations over 10 iterations
    np_time = timeit.timeit(
        lambda: generate_signals_np(condition_buy, condition_sell), number=10
    )
    # np_time = timeit.timeit(
    #     lambda: main(), number=10
    # )
    numba_time = timeit.timeit(
        lambda: generate_signals_numba(condition_buy, condition_sell), number=10
    )

    print("Time for np.select version (10 iterations): {:.6f} seconds".format(np_time))
    print("Time for numba version (10 iterations): {:.6f} seconds".format(numba_time))
    print(f"Speedup factor: {np_time/numba_time:.2f}x")

    return signals_numba


def test_signal_performance2():
    # tickers_list = ["AAPL"]  # test
    ticker = "AAPL"  # test
    con_price_data = sqlite3.connect(PRICE_DB_PATH)
    ticker_price_history = pd.read_sql_query(
        "SELECT * FROM '{tab}'".format(tab=ticker),
        con_price_data,
        index_col="Date",
    )
    con_price_data.close()

    # strategy_result = strategy(ticker_price_history.copy())
    def run_strats_np():
        strategy_result = BBANDS_indicator(ticker_price_history.copy())
        strategy_result = BBANDS_indicator(ticker_price_history.copy())
        strategy_result = BBANDS_indicator(ticker_price_history.copy())
        strategy_result = BBANDS_indicator(ticker_price_history.copy())
        strategy_result = BBANDS_indicator(ticker_price_history.copy())
        strategy_result = DEMA_indicator(ticker_price_history.copy())
        strategy_result = DEMA_indicator(ticker_price_history.copy())
        strategy_result = DEMA_indicator(ticker_price_history.copy())
        strategy_result = DEMA_indicator(ticker_price_history.copy())
        strategy_result = DEMA_indicator(ticker_price_history.copy())

    def run_strats_numba():
        strategy_result = BBANDS_indicator_numba(ticker_price_history.copy())
        strategy_result = BBANDS_indicator_numba(ticker_price_history.copy())
        strategy_result = BBANDS_indicator_numba(ticker_price_history.copy())
        strategy_result = BBANDS_indicator_numba(ticker_price_history.copy())
        strategy_result = BBANDS_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())
        strategy_result = DEMA_indicator_numba(ticker_price_history.copy())

    numba_time = timeit.timeit(lambda: run_strats_numba(), number=10)
    np_time = timeit.timeit(lambda: run_strats_np(), number=10)

    print("Time for np.select version (10 iterations): {:.6f} seconds".format(np_time))
    print("Time for numba version (10 iterations): {:.6f} seconds".format(numba_time))
    print(f"Speedup factor: {np_time/numba_time:.2f}x")

    return


# --- Overlap Studies ---
def BBANDS_indicator_numba(data, timeperiod=20):
    """Vectorized Bollinger Bands (BBANDS) indicator signals."""
    upper, middle, lower = ta.BBANDS(data["Close"], timeperiod=timeperiod)
    data["BBANDS_indicator"] = generate_signals_numba(
        condition_buy=data["Close"].values < lower.values,  # Use .values
        condition_sell=data["Close"].values > upper.values,  # Use .values
    )
    return data["BBANDS_indicator"]


def DEMA_indicator_numba(data, timeperiod=30):
    """Vectorized Double Exponential Moving Average (DEMA) indicator signals."""
    dema = ta.DEMA(data["Close"], timeperiod=timeperiod)
    data["DEMA_indicator"] = generate_signals_numba(
        condition_buy=data["Close"].values > dema.values,  # Use .values
        condition_sell=data["Close"].values < dema.values,  # Use .values
    )
    return data["DEMA_indicator"]


def BBANDS_indicator(data, timeperiod=20):
    """Vectorized Bollinger Bands (BBANDS) indicator signals."""
    upper, middle, lower = ta.BBANDS(data["Close"], timeperiod=timeperiod)
    data["BBANDS_indicator"] = generate_signals_np(
        condition_buy=data["Close"] < lower,
        condition_sell=data["Close"] > upper,
    )
    return data["BBANDS_indicator"]


def DEMA_indicator(data, timeperiod=30):
    """Vectorized Double Exponential Moving Average (DEMA) indicator signals."""
    dema = ta.DEMA(data["Close"], timeperiod=timeperiod)
    data["DEMA_indicator"] = generate_signals_np(
        condition_buy=data["Close"] > dema,
        condition_sell=data["Close"] < dema,
    )
    return data["DEMA_indicator"]


# Example usage:
if __name__ == "__main__":
    # Create synthetic data for testing
    np.random.seed(0)
    size = 1000000  # using one million entries for a robust test

    # Prepare test data
    data = {"Close": np.random.random(size)}
    lower_threshold = 0.3
    upper_threshold = 0.7

    condition_buy = data["Close"] < lower_threshold
    condition_sell = data["Close"] > upper_threshold

    # Run performance test
    # test_signal_performance(condition_buy, condition_sell)
    test_signal_performance2()
