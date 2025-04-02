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
    test_period_end,
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


def create_position_column_vectorized(df):
    """
    Creates a 'position' column in the DataFrame based on the 'Action' values.

    The position is set to 1 when 'Action' changes from 'Hold' or 'Sell' to 'Buy'.
    It remains 1 until 'Action' changes from 'Buy' to 'Sell' or from 'Hold' to 'Sell',
    at which point it reverts to the default value of 0.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'Action' column

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with an additional 'position' column
    """
    import numpy as np
    import pandas as pd

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Create a shifted version of Action to detect changes
    df["prev_action"] = df["Action"].shift(1)

    # Initialize position column with zeros
    df["position"] = 0

    # Define conditions for setting position to 1
    # Condition 1: Action changes from Hold or Sell to Buy
    buy_signal = (df["prev_action"].isin(["Hold", "Sell"])) & (df["Action"] == "Buy")

    # Condition 2: Action changes from Buy to Sell or from Hold to Sell
    sell_signal = (df["prev_action"].isin(["Buy", "Hold"])) & (df["Action"] == "Sell")

    # Create a mask to track position state
    position_mask = np.zeros(len(df), dtype=int)

    # Use numpy's where to vectorize the state tracking
    # When buy_signal is True, set position to 1
    # When sell_signal is True, set position to 0
    # Otherwise, keep the previous position value

    # Convert boolean arrays to numpy arrays for faster processing
    buy_signal_array = buy_signal.to_numpy()
    sell_signal_array = sell_signal.to_numpy()

    # Iterate through the array once to build the position state
    for i in range(1, len(df)):
        if buy_signal_array[i - 1]:
            position_mask[i] = 1
        elif sell_signal_array[i - 1]:
            position_mask[i] = 0
        else:
            position_mask[i] = position_mask[i - 1]

    # Assign the calculated position to the DataFrame
    df["position"] = position_mask

    # Drop the temporary column
    df.drop("prev_action", axis=1, inplace=True)

    return df


def create_cash_holdings_column(df):
    """
    Creates 'cash' and 'holdings' columns in the DataFrame based on position changes.

    The cash starts at 10000. When position changes from 0 to 1, the Close price
    is subtracted from cash. When position changes from 1 to 0, the Close price
    is added to cash.

    The holdings start at 0. When position changes from 0 to 1, add Close price to holdings
    and adjust it each day based on Close. When position changes from 1 to 0, holdings
    reverts to 0.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'position' and 'Close' columns

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with additional 'cash' and 'holdings' columns
    """
    import numpy as np
    import pandas as pd

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Create a shifted version of position to detect changes
    df["prev_position"] = df["position"].shift(1).fillna(0)

    # Identify buy signals (position changes from 0 to 1)
    buy_signals = (df["prev_position"] == 0) & (df["position"] == 1)

    # Identify sell signals (position changes from 1 to 0)
    sell_signals = (df["prev_position"] == 1) & (df["position"] == 0)

    # Initialize cash and holdings columns with float dtype to avoid warnings
    df["cash"] = 10000.0
    df["holdings"] = 0.0

    # Calculate cash changes
    cash_changes = pd.Series(0.0, index=df.index)
    cash_changes.loc[buy_signals] = -df.loc[buy_signals, "Close"].astype(float)
    cash_changes.loc[sell_signals] = df.loc[sell_signals, "Close"].astype(float)

    # Apply cash changes cumulatively
    for i in range(1, len(df)):
        df.loc[df.index[i], "cash"] = (
            df.loc[df.index[i - 1], "cash"] + cash_changes.iloc[i]
        )

    # For holdings, we need to track when we're in a position and update the value daily
    # Initialize a temporary column to track when we're in a position
    df["in_position"] = 0

    # Set up the in_position column (1 when we're holding, 0 when we're not)
    for i in range(1, len(df)):
        if buy_signals.iloc[i]:
            df.loc[df.index[i], "in_position"] = 1
        elif sell_signals.iloc[i]:
            df.loc[df.index[i], "in_position"] = 0
        else:
            df.loc[df.index[i], "in_position"] = df.loc[df.index[i - 1], "in_position"]

    # Calculate holdings based on position and Close price
    # When in position, holdings = current Close price
    # When not in position, holdings = 0
    df["holdings"] = df["in_position"] * df["Close"]
    df["total"] = df["holdings"] + df["cash"]

    # Drop the temporary columns
    df.drop(["prev_position", "in_position"], axis=1, inplace=True)

    return df


def create_buy_and_sell_date_column(df):
    """
    Creates 'buy_date' and 'sell_date' columns in the DataFrame based on position changes.

    The buy_date starts as an empty string. When position changes from 0 to 1,
    the index date is copied into the buy_date column. This buy_date is repeated
    until position changes from 1 to 0, at which point buy_date is set to empty string.

    The sell_date column is populated with the index date when position changes from 1 to 0.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'position' column

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with additional 'buy_date' and 'sell_date' columns
    """
    import pandas as pd

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Create a shifted version of position to detect changes
    df["prev_position"] = df["position"].shift(1).fillna(0)

    # Identify buy signals (position changes from 0 to 1)
    buy_signals = (df["prev_position"] == 0) & (df["position"] == 1)

    # Identify sell signals (position changes from 1 to 0)
    sell_signals = (df["prev_position"] == 1) & (df["position"] == 0)

    # Initialize buy_date and sell_date columns with empty strings
    df["buy_date"] = ""
    df["sell_date"] = ""

    # Set buy_date when buy signal occurs (position changes from 0 to 1)
    # and set sell_date when sell signal occurs (position changes from 1 to 0)
    for i in range(len(df)):
        if buy_signals.iloc[i]:
            df.loc[df.index[i], "buy_date"] = df.index[i]
            # df.loc[df.index[i], "buy_price"] = df.loc[df.index[i], "Close"]
        elif sell_signals.iloc[i]:
            df.loc[df.index[i], "sell_date"] = df.index[i]
            # df.loc[df.index[i], "sell_price"] = df.loc[df.index[i], "Close"]
            df.loc[df.index[i], "buy_date"] = df.loc[df.index[i - 1], "buy_date"]
            # df.loc[df.index[i], "buy_price"] = df.loc[df.index[i - 1], "buy_price"]
        elif i > 0 and df["position"].iloc[i] == 1:
            # Propagate the buy_date when in position
            df.loc[df.index[i], "buy_date"] = df.loc[df.index[i - 1], "buy_date"]
            # df.loc[df.index[i], "buy_price"] = df.loc[df.index[i - 1], "buy_price"]

    # Drop the temporary column
    df.drop("prev_position", axis=1, inplace=True)

    return df


def lookup_price_data(trades_df, price_conn):
    """
    Efficiently merges trade data with price data using pandas vectorized operations.

    This function:
    1. Reads trades list from trades_df
    2. For each ticker in the trades list:
       a. Reads price data from 'price_data.db' (stored as one table per ticker)
       b. Looks up 'Close' prices for buy_date and sell_date
       c. Adds these prices to the trades dataframe as 'buy_price' and 'sell_price'

    Returns:
        pandas.DataFrame: Trades dataframe with added buy_price and sell_price columns
    """

    # Connect to trades database and read the trades list
    # trades_conn = sqlite3.connect('trades_list_vectorized.db')
    # trades_df = pd.read_sql_query("SELECT * FROM trades_list", trades_conn)
    # trades_conn.close()

    # Ensure date columns are in datetime format
    trades_df["buy_date"] = pd.to_datetime(trades_df["buy_date"])
    trades_df["sell_date"] = pd.to_datetime(trades_df["sell_date"])

    # Connect to price database
    # price_conn = sqlite3.connect(PRICE_DB_PATH)

    # Get unique tickers from trades list to minimize database queries
    unique_tickers = trades_df["Ticker"].unique()

    # Create a dictionary to store price data for each ticker
    price_data = {}

    # For each unique ticker, fetch price data and store in dictionary
    for ticker in unique_tickers:
        try:
            # Get all dates needed for this ticker (both buy and sell dates)
            ticker_trades = trades_df[trades_df["Ticker"] == ticker]
            needed_dates = pd.concat(
                [
                    pd.DataFrame({"date": ticker_trades["buy_date"]}),
                    pd.DataFrame({"date": ticker_trades["sell_date"]}),
                ]
            ).drop_duplicates()

            # Format dates for SQL query
            date_strings = "', '".join(needed_dates["date"].dt.strftime("%Y-%m-%d"))

            # Query only the necessary dates for this ticker
            query = (
                f"SELECT Date, Close FROM '{ticker}' WHERE Date IN ('{date_strings}')"
            )
            ticker_prices = pd.read_sql_query(query, price_conn)

            # Convert Date to datetime for merging
            ticker_prices["Date"] = pd.to_datetime(ticker_prices["Date"])

            # Store in dictionary for quick lookup
            price_data[ticker] = ticker_prices.set_index("Date")["Close"]

        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")

    # Create empty columns for buy and sell prices
    trades_df["buy_price"] = None
    trades_df["sell_price"] = None

    # Use vectorized operations to look up prices for each ticker group
    for ticker in unique_tickers:
        ticker_mask = trades_df["Ticker"] == ticker
        if ticker in price_data:
            # Look up buy prices
            trades_df.loc[ticker_mask, "buy_price"] = trades_df.loc[
                ticker_mask, "buy_date"
            ].map(price_data[ticker])

            # Look up sell prices
            trades_df.loc[ticker_mask, "sell_price"] = trades_df.loc[
                ticker_mask, "sell_date"
            ].map(price_data[ticker])

    trades_df["buy_date"] = trades_df["buy_date"].dt.strftime("%Y-%m-%d")
    trades_df["sell_date"] = trades_df["sell_date"].dt.strftime("%Y-%m-%d")

    return trades_df


def prepare_sp500_one_day_return(conn):
    """
    Efficiently calculates the one-day percentage return for S&P 500 data and updates the database.

    This function:
    1. Retrieves S&P 500 price data from the '^GSPC' table in the SQLite database
    2. Calculates the one-day percentage return using pandas vectorized operations
    3. Saves the updated dataframe back to the database, replacing the original table

    Args:
        conn: SQLite database connection to 'price_data.db'

    Returns:
        pandas.DataFrame: The updated S&P 500 dataframe with the '1_day_pct_return' column
    """
    import pandas as pd

    # Read S&P 500 data from the database
    query = "SELECT * FROM '^GSPC'"
    sp500_df = pd.read_sql_query(query, conn)

    # Ensure Date column is in datetime format
    if "Date" in sp500_df.columns:
        sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])

    # sp500_df.drop(columns=["1_day_spy_return"], inplace=True)
    # Calculate one-day percentage return using vectorized operations
    sp500_df["One_day_spy_return"] = sp500_df["Close"].pct_change().round(4) * 100

    # Replace NaN values with 0 for the first row
    sp500_df["One_day_spy_return"] = sp500_df["One_day_spy_return"].fillna(0)

    # Save the updated dataframe back to the database
    # First, drop the existing table
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS '^GSPC'")
    conn.commit()

    sp500_df["Date"] = sp500_df["Date"].dt.strftime("%Y-%m-%d")
    # Then write the updated dataframe to a new table with the same name
    sp500_df.to_sql(
        "^GSPC",
        conn,
        index=False,
        if_exists="replace",
        dtype={"Date": "TEXT PRIMARY KEY NOT NULL"},
    )

    return sp500_df


def lookup_regime_data(trades_df, price_conn):
    """
    Looks up regime data (VIX Close price and S&P500 one-day return) for trade buy dates.

    Args:
        trades_df (pd.DataFrame): DataFrame containing trades with columns 'Ticker', 'buy_date', 'sell_date'
        price_conn (sqlite3.Connection): Connection to price_data.db SQLite database

    Returns:
        pd.DataFrame: Original trades_df with added columns '^VIX' and 'One_day_spy_return'
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = trades_df.copy()

    # Convert buy_date to datetime if not already
    result_df["buy_date"] = pd.to_datetime(result_df["buy_date"])

    # Get unique buy dates to minimize database queries
    unique_dates = result_df["buy_date"].dt.strftime("%Y-%m-%d").unique()

    # Query VIX data for all unique dates at once
    vix_query = f"""
    SELECT Date, Close 
    FROM "^VIX" 
    WHERE Date IN ({','.join(['?']*len(unique_dates))})
    """
    vix_data = pd.read_sql_query(vix_query, price_conn, params=tuple(unique_dates))

    # Query S&P500 data for all unique dates at once
    spy_query = f"""
    SELECT Date, One_day_spy_return 
    FROM "^GSPC" 
    WHERE Date IN ({','.join(['?']*len(unique_dates))})
    """
    spy_data = pd.read_sql_query(spy_query, price_conn, params=tuple(unique_dates))

    # Check if we got data back
    if vix_data.empty:
        print("Warning: No VIX data found for the specified dates")
    if spy_data.empty:
        print("Warning: No S&P500 data found for the specified dates")

    # Convert date columns to datetime for proper merging
    vix_data["Date"] = pd.to_datetime(vix_data["Date"])
    spy_data["Date"] = pd.to_datetime(spy_data["Date"])

    # Convert buy_date to string format for merging
    result_df["date_key"] = result_df["buy_date"].dt.strftime("%Y-%m-%d")

    # Merge VIX data
    if not vix_data.empty:
        vix_data["date_key"] = vix_data["Date"].dt.strftime("%Y-%m-%d")
        result_df = pd.merge(
            result_df, vix_data[["date_key", "Close"]], on="date_key", how="left"
        )
        result_df.rename(columns={"Close": "^VIX"}, inplace=True)
    else:
        result_df["^VIX"] = None

    # Merge S&P500 data
    if not spy_data.empty:
        spy_data["date_key"] = spy_data["Date"].dt.strftime("%Y-%m-%d")
        result_df = pd.merge(
            result_df,
            spy_data[["date_key", "One_day_spy_return"]],
            on="date_key",
            how="left",
        )
    else:
        result_df["One_day_spy_return"] = None

    # Drop the temporary key column
    result_df.drop("date_key", axis=1, inplace=True)

    # Check for missing values after merge
    missing_vix = result_df["^VIX"].isna().sum()
    missing_spy = result_df["One_day_spy_return"].isna().sum()

    if missing_vix > 0:
        print(f"Warning: {missing_vix} rows have missing VIX data")
    if missing_spy > 0:
        print(f"Warning: {missing_spy} rows have missing S&P500 data")

    result_df["buy_date"] = result_df["buy_date"].dt.strftime("%Y-%m-%d")

    return result_df


if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    logger = setup_logging("logs", "trades_list.log", level=logging.WARNING)
    """
    create trades list from strategy decisions
    """
    # strategies = strategies_test
    # strategies = [strategies_test[3]]
    tickers_list = train_tickers + regime_tickers
    logger.info(f"=== START COMPUTE TRADES LIST ===")
    logger.info(f"{len(train_tickers) = } {len(regime_tickers) = }\n")

    # setup db connections
    price_data_dir = "PriceData"
    strategy_decisions_db_name = os.path.join(
        price_data_dir, "strategy_decisions_final.db"
    )
    con_sd = sqlite3.connect(strategy_decisions_db_name)
    trades_list_db_name = os.path.join(price_data_dir, "trades_list_vectorised.db")
    con_tl = sqlite3.connect(trades_list_db_name)
    con_pd = sqlite3.connect(PRICE_DB_PATH)

    start_date = datetime.strptime(train_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(test_period_end, "%Y-%m-%d")

    # Subtract 1 business day from start_date - needed for sp500 1 day return.
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

    # === prepare REGIME ma calcs eg 1-day spy return. Use pandas dataframe.
    logger.info(f"prepare regime data")
    prepare_sp500_one_day_return(con_pd)
    # ticker_price_history = prepare_regime_data(ticker_price_history, logger)
    # logger.info(f"{ticker_price_history = }")

    # 3. run training
    logger.info(f"Training period: {start_date} to {end_date}")
    # strategies = [strategies[0]]
    # ticker = "AAPL"
    # ticker_list = ["MSFT", "AAPL", "ARM"]
    ticker_list = train_tickers
    for idx, strategy in enumerate(strategies):
        strategy_name = strategy.__name__
        logger.info(
            f"\n=== COMPUTING TRADES FOR: {strategy_name} ({idx + 1}/{len(strategies)}) ==="
        )

        df = sql_to_df_with_date_range(strategy_name, start_date, end_date, con_sd)

        # Melt df_existing to stack ticker columns to rows
        df = df.reset_index().melt(
            id_vars=["Date"], var_name="Ticker", value_name="Action"
        )
        # logger.info(f"{df = }")
        # df_trades_dict = {}
        df_trades_list = []

        for ticker in ticker_list:

            df_ticker = df[df["Ticker"] == ticker]
            df_ticker = df_ticker.set_index("Date")

            df_ticker = create_position_column_vectorized(df_ticker)
            # df_ticker = create_cash_holdings_column(df_ticker)
            df_ticker = create_buy_and_sell_date_column(df_ticker)

            cols_to_Keep2 = ["Ticker", "buy_date", "sell_date"]
            df_trades = df_ticker[cols_to_Keep2][df_ticker["sell_date"] != ""]

            logger.info(f"{ticker} {len(df_trades) = }")

            # df_trades_dict[ticker] = df_trades
            df_trades_list.append(df_trades)

        # append all df_trades into 1 dataframe
        trades_df = pd.concat(df_trades_list)
        number_of_trades = len(trades_df)
        logger.info(f"{strategy_name} {number_of_trades = }")
        if number_of_trades > 0:
            # get/merge price and regime data
            logger.info(f"lookup prices...")

            trades_df = lookup_price_data(trades_df, con_pd)

            trades_df = lookup_regime_data(trades_df, con_pd)

            trades_df["ratio"] = trades_df["sell_price"] / trades_df["buy_price"]

            # df_trades["return"] = df_trades["ratio"] - 1
            # df_trades["cum_return"] = df_trades["return"].cumsum()

            trades_df["trade_id"] = trades_df["buy_date"] + trades_df["Ticker"]
            trades_df.set_index("trade_id", inplace=True)
            trades_df.sort_index(inplace=True)

            """
            How to merge new data/trades? eg for latest dates?
            """

            trades_df.to_sql(
                strategy_name,
                con_tl,
                if_exists="replace",
                index=True,
                dtype={"trade_id": "TEXT PRIMARY KEY"},
            )

    con_pd.close()
    con_tl.close()
    con_sd.close()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")

    logger.info(f"\n\n=== SUMMARY ===\n")
    # logger.info(f"{number_of_trades_by_strategy = }")
    logger.info(f"Finished!")
