import functools
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import os
import sys

import pandas as pd
import yfinance as yf

# Ensure sys.path manipulation is at the top, before other local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import (
    trade_asset_limit,
    train_loss_price_change_ratio_d1,
    train_loss_price_change_ratio_d2,
    train_loss_profit_time_d1,
    train_loss_profit_time_d2,
    train_loss_profit_time_else,
    train_profit_price_change_ratio_d1,
    train_profit_price_change_ratio_d2,
    train_profit_profit_time_d1,
    train_profit_profit_time_d2,
    train_profit_profit_time_else,
    train_rank_asset_limit,
    train_rank_liquidity_limit,
    train_time_delta_balanced,
    train_time_delta_increment,
    train_time_delta_multiplicative,
    mode,
)
from helper_files.client_helper import (
    get_ndaq_tickers,
    strategies,
    store_dict_as_json,
    save_df_to_csv,
    load_json_to_dict,
)
from helper_files.train_client_helper import get_historical_data

from PriceData.store_price_data import sql_to_df_with_date_range

# from strategies.talib_indicators import *


def initialize_simulation(
    period_start,
    period_end,
    train_tickers,
    mongo_client,
    FINANCIAL_PREP_API_KEY,
    logger,
):
    """
    Initializes the simulation by loading necessary data and setting up initial states.
    Optimizations:
      - Batch fetch indicator periods from MongoDB.
      - Bulk download ticker historical data using yfinance with threading.
      - Fallback to individual max period download if bulk download returns no data.
    """
    logger.info("Initializing simulation...")

    ticker_price_history = {}
    ideal_period = {}

    def get_ideal_period():
        # Connect to MongoDB and batch fetch indicator periods for all strategies.
        db = mongo_client.IndicatorsDatabase
        indicator_collection = db.Indicators
        logger.info("Connected to MongoDB: Retrieved Indicators collection.")

        # Assuming 'strategies' is a global list of strategy objects
        strategy_names = [strategy.__name__ for strategy in strategies]
        indicator_docs = list(
            indicator_collection.find({"indicator": {"$in": strategy_names}})
        )
        indicator_lookup = {
            doc["indicator"]: doc.get("ideal_period") for doc in indicator_docs
        }
        for strategy in strategies:
            if strategy.__name__ in indicator_lookup:
                ideal_period[strategy.__name__] = indicator_lookup[strategy.__name__]
                # logger.info(f"Retrieved ideal period for {strategy.__name__}: {indicator_lookup[strategy.__name__]}")
            else:
                logger.info(
                    f"No ideal period found for {strategy.__name__}, using default."
                )
        return ideal_period

    # load/save local copy of ideal_period
    ideal_period_dir = "results"
    ideal_period_filename = f"ideal_period.json"
    ideal_period_filepath = os.path.join(ideal_period_dir, ideal_period_filename)
    if not os.path.exists(ideal_period_filepath):
        ideal_period = get_ideal_period()
        store_dict_as_json(
            ideal_period, ideal_period_filename, ideal_period_dir, logger
        )
    else:
        ideal_period = load_json_to_dict(ideal_period_dir, ideal_period_filename)

    # If no tickers provided, fetch Nasdaq tickers
    if not train_tickers:
        logger.info("No tickers provided. Fetching Nasdaq tickers...")
        train_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
        logger.info(f"Fetched {len(train_tickers)} tickers.")

    # Determine the historical data start date (2 years before period_start)
    start_date = datetime.strptime(period_start, "%Y-%m-%d")

    # adjust start date to 1 day previous to avoid lookahead bias. 2 days for 1day pct change for sp500
    start_date -= timedelta(days=2)

    data_start_date = (start_date - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"Fetching historical data from {data_start_date} to {period_end}.")

    def get_historical_data_from_yfinance():
        # Bulk download ticker data using yfinance (with threading enabled)
        try:
            bulk_data = yf.download(
                tickers=train_tickers,
                start=data_start_date,
                end=period_end,
                interval="1d",
                group_by="ticker",
                threads=True,
            )
        except Exception as bulk_err:
            logger.error(f"Bulk download failed: {str(bulk_err)}")
            bulk_data = None

        # Process each ticker's data. If bulk data is missing or empty for a ticker, fall back to max period download.
        for ticker in train_tickers:
            try:
                ticker_data = None
                if bulk_data is not None:
                    # If multiple tickers, data has a MultiIndex on columns.
                    if (
                        isinstance(bulk_data.columns, pd.MultiIndex)
                        and ticker in bulk_data.columns.levels[0]
                    ):
                        ticker_data = bulk_data[ticker]
                    # If only one ticker was downloaded, bulk_data may be a regular DataFrame.
                    elif not isinstance(bulk_data.columns, pd.MultiIndex):
                        ticker_data = bulk_data
                # Check if data was successfully retrieved.
                if ticker_data is None or ticker_data.empty:
                    raise Exception("No data from bulk download")
                # logger.info(f"Successfully retrieved data for {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()}")
                ticker_price_history[ticker] = ticker_data
            except Exception as e:
                logger.info(
                    f"Error retrieving specific date range for {ticker} via bulk download, fetching max available data. Error: {str(e)}"
                )
                try:
                    ticker_data = yf.Ticker(ticker).history(period="max", interval="1d")
                    if ticker_data.empty:
                        logger.warning(
                            f"No data available for {ticker} even for max period."
                        )
                    else:
                        logger.info(
                            f"Successfully retrieved max available data for {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()}"
                        )
                    ticker_price_history[ticker] = ticker_data
                except Exception as e2:
                    logger.error(f"Failed to retrieve data for {ticker}: {str(e2)}")
                    ticker_price_history[ticker] = None

    price_data_source = "db"

    if price_data_source == "yfinance":
        logger.info("Fetching historical data from yfinance.")
        ticker_price_history = get_historical_data_from_yfinance()

    elif price_data_source == "db":
        logger.info("Fetching historical data from database.")
        for ticker in train_tickers:
            ticker_price_history[ticker] = sql_to_df_with_date_range(
                ticker, data_start_date, period_end
            )

    logger.info("Simulation initialization complete.")
    return ticker_price_history, ideal_period


def initialize_simulation_old(
    period_start,
    period_end,
    train_tickers,
    mongo_client,
    FINANCIAL_PREP_API_KEY,
    logger,
):
    """
    Initializes the simulation by loading necessary data and setting up initial states.
    Optimizations:
      - Batch fetch indicator periods from MongoDB.
      - Bulk download ticker historical data using yfinance with threading.
      - Fallback to individual max period download if bulk download returns no data.
    """
    logger.info("Initializing simulation...")

    ticker_price_history = {}
    ideal_period = {}

    # Connect to MongoDB and batch fetch indicator periods for all strategies.
    db = mongo_client.IndicatorsDatabase
    indicator_collection = db.Indicators
    logger.info("Connected to MongoDB: Retrieved Indicators collection.")

    # Assuming 'strategies' is a global list of strategy objects
    strategy_names = [strategy.__name__ for strategy in strategies]
    indicator_docs = list(
        indicator_collection.find({"indicator": {"$in": strategy_names}})
    )
    indicator_lookup = {
        doc["indicator"]: doc.get("ideal_period") for doc in indicator_docs
    }
    for strategy in strategies:
        if strategy.__name__ in indicator_lookup:
            ideal_period[strategy.__name__] = indicator_lookup[strategy.__name__]
            # logger.info(f"Retrieved ideal period for {strategy.__name__}: {indicator_lookup[strategy.__name__]}")
        else:
            logger.info(
                f"No ideal period found for {strategy.__name__}, using default."
            )

    # If no tickers provided, fetch Nasdaq tickers
    if not train_tickers:
        logger.info("No tickers provided. Fetching Nasdaq tickers...")
        train_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
        logger.info(f"Fetched {len(train_tickers)} tickers.")

    # Determine the historical data start date (2 years before period_start)
    start_date = datetime.strptime(period_start, "%Y-%m-%d")

    # adjust start date to 1 day previous to avoid lookahead bias. 2 days for 1day pct change for sp500
    start_date -= timedelta(days=2)

    data_start_date = (start_date - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"Fetching historical data from {data_start_date} to {period_end}.")

    # Bulk download ticker data using yfinance (with threading enabled)
    try:
        bulk_data = yf.download(
            tickers=train_tickers,
            start=data_start_date,
            end=period_end,
            interval="1d",
            group_by="ticker",
            threads=True,
        )
    except Exception as bulk_err:
        logger.error(f"Bulk download failed: {str(bulk_err)}")
        bulk_data = None

    # Process each ticker's data. If bulk data is missing or empty for a ticker, fall back to max period download.
    for ticker in train_tickers:
        try:
            ticker_data = None
            if bulk_data is not None:
                # If multiple tickers, data has a MultiIndex on columns.
                if (
                    isinstance(bulk_data.columns, pd.MultiIndex)
                    and ticker in bulk_data.columns.levels[0]
                ):
                    ticker_data = bulk_data[ticker]
                # If only one ticker was downloaded, bulk_data may be a regular DataFrame.
                elif not isinstance(bulk_data.columns, pd.MultiIndex):
                    ticker_data = bulk_data
            # Check if data was successfully retrieved.
            if ticker_data is None or ticker_data.empty:
                raise Exception("No data from bulk download")
            # logger.info(f"Successfully retrieved data for {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()}")
            ticker_price_history[ticker] = ticker_data
        except Exception as e:
            logger.info(
                f"Error retrieving specific date range for {ticker} via bulk download, fetching max available data. Error: {str(e)}"
            )
            try:
                ticker_data = yf.Ticker(ticker).history(period="max", interval="1d")
                if ticker_data.empty:
                    logger.warning(
                        f"No data available for {ticker} even for max period."
                    )
                else:
                    logger.info(
                        f"Successfully retrieved max available data for {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()}"
                    )
                ticker_price_history[ticker] = ticker_data
            except Exception as e2:
                logger.error(f"Failed to retrieve data for {ticker}: {str(e2)}")
                ticker_price_history[ticker] = None

    logger.info("Simulation initialization complete.")
    return ticker_price_history, ideal_period


def update_time_delta(time_delta, mode):
    """
    Updates time_delta based on the specified mode
    """
    if mode == "additive":
        return time_delta + train_time_delta_increment
    elif mode == "multiplicative":
        return time_delta * train_time_delta_multiplicative
    elif mode == "balanced":
        return time_delta + train_time_delta_balanced * time_delta
    return time_delta


def update_points_and_trades(
    strategy,
    ratio,
    current_price,
    trading_simulator,
    points,
    time_delta,
    ticker,
    qty,
    current_date,
):
    """
    Updates points based on trade performance and manages trade statistics.

    Stores all completed trades in trading_simulator.
    """
    # New log trade
    date_str = current_date.strftime("%Y-%m-%d")

    trade_details = [
        strategy.__name__,
        ticker,
        round(current_price, 2),
        round(trading_simulator[strategy.__name__]["holdings"][ticker]["price"], 2),
        qty,
        round(ratio, 4),
        round(trading_simulator[strategy.__name__]["holdings"][ticker]["vix"], 2),
        round(
            trading_simulator[strategy.__name__]["holdings"][ticker][
                "1day_sp500_return"
            ]
            * 100,
            2,
        ),
        trading_simulator[strategy.__name__]["holdings"][ticker]["buy_date"],
        date_str,
    ]
    trading_simulator[strategy.__name__]["trades_list"].append(trade_details)
    # new end

    if (
        current_price
        > trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
    ):
        trading_simulator[strategy.__name__]["successful_trades"] += 1
        if ratio < train_profit_price_change_ratio_d1:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_d1
            )
        elif ratio < train_profit_price_change_ratio_d2:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_d2
            )
        else:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_else
            )
    elif (
        current_price
        == trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
    ):
        trading_simulator[strategy.__name__]["neutral_trades"] += 1
    else:
        trading_simulator[strategy.__name__]["failed_trades"] += 1
        if ratio > train_loss_price_change_ratio_d1:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_d1
            )
        elif ratio > train_loss_price_change_ratio_d2:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_d2
            )
        else:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_else
            )

    trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] -= qty
    if trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] == 0:
        del trading_simulator[strategy.__name__]["holdings"][ticker]
    elif trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] < 0:
        raise Exception("Quantity cannot be negative")
    trading_simulator[strategy.__name__]["total_trades"] += 1

    return points, trading_simulator


def execute_trade(
    decision,
    qty,
    ticker,
    current_price,
    strategy,
    trading_simulator,
    points,
    time_delta,
    portfolio_qty,
    total_portfolio_value,
    current_regime_data,
    current_date,
    logger,
):
    """
    Executes a trade based on the strategy decision and updates trading simulator and points
    """
    if (
        decision == "buy"
        and trading_simulator[strategy.__name__]["amount_cash"]
        > train_rank_liquidity_limit
        and qty > 0
        and ((portfolio_qty + qty) * current_price) / total_portfolio_value
        < train_rank_asset_limit
    ):
        logger.debug(
            f"debug: buy {qty} {ticker} at {current_price}, {current_date} {strategy.__name__}"
        )
        trading_simulator[strategy.__name__]["amount_cash"] -= qty * current_price

        if ticker in trading_simulator[strategy.__name__]["holdings"]:
            trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] += qty
        else:
            trading_simulator[strategy.__name__]["holdings"][ticker] = {"quantity": qty}

        trading_simulator[strategy.__name__]["holdings"][ticker][
            "price"
        ] = current_price
        trading_simulator[strategy.__name__]["total_trades"] += 1

        # new add buy date
        date_str = current_date.strftime("%Y-%m-%d")
        trading_simulator[strategy.__name__]["holdings"][ticker]["buy_date"] = date_str
        trading_simulator[strategy.__name__]["holdings"][ticker]["vix"] = (
            current_regime_data[0]
        )
        trading_simulator[strategy.__name__]["holdings"][ticker][
            "1day_sp500_return"
        ] = current_regime_data[1]
        # new end

    elif (
        decision == "sell"
        and trading_simulator[strategy.__name__]["holdings"]
        .get(ticker, {})
        .get("quantity", 0)
        >= qty
    ):
        trading_simulator[strategy.__name__]["amount_cash"] += qty * current_price
        ratio = (
            current_price
            / trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
        )

        points, trading_simulator = update_points_and_trades(
            strategy,
            ratio,
            current_price,
            trading_simulator,
            points,
            time_delta,
            ticker,
            qty,
            current_date,
        )

    return trading_simulator, points


def execute_trade_only_buy_one(
    decision,
    qty,
    ticker,
    current_price,
    strategy,
    trading_simulator,
    points,
    time_delta,
    portfolio_qty,
    total_portfolio_value,
    current_regime_data,
    current_date,
    logger,
):
    """
    Executes a trade based on the strategy decision and updates trading simulator and points
    """
    if (
        decision
        == "buy"
        # and trading_simulator[strategy.__name__]["amount_cash"]
        # > train_rank_liquidity_limit
        # and qty > 0
        # and ((portfolio_qty + qty) * current_price) / total_portfolio_value
        # < train_rank_asset_limit
    ):
        logger.debug(
            f"debug: buy {qty} {ticker} at {current_price}, {current_date} {strategy.__name__}"
        )
        trading_simulator[strategy.__name__]["amount_cash"] -= qty * current_price

        if ticker in trading_simulator[strategy.__name__]["holdings"]:
            trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] += qty
        else:
            trading_simulator[strategy.__name__]["holdings"][ticker] = {"quantity": qty}

        trading_simulator[strategy.__name__]["holdings"][ticker][
            "price"
        ] = current_price
        trading_simulator[strategy.__name__]["total_trades"] += 1

        # new add buy date
        date_str = current_date.strftime("%Y-%m-%d")
        trading_simulator[strategy.__name__]["holdings"][ticker]["buy_date"] = date_str
        trading_simulator[strategy.__name__]["holdings"][ticker]["vix"] = (
            current_regime_data[0]
        )
        trading_simulator[strategy.__name__]["holdings"][ticker][
            "1day_sp500_return"
        ] = current_regime_data[1]
        # new end

    elif (
        decision
        == "sell"
        # and trading_simulator[strategy.__name__]["holdings"]
        # .get(ticker, {})
        # .get("quantity", 0)
        # >= qty
    ):
        trading_simulator[strategy.__name__]["amount_cash"] += qty * current_price
        ratio = (
            current_price
            / trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
        )

        points, trading_simulator = update_points_and_trades(
            strategy,
            ratio,
            current_price,
            trading_simulator,
            points,
            time_delta,
            ticker,
            qty,
            current_date,
        )

    return trading_simulator, points


def simulate_trading_day(
    current_date,
    strategies,
    trading_simulator,
    points,
    time_delta,
    ticker_price_history,
    train_tickers,
    precomputed_decisions,
    logger,
    regime_tickers,
):
    """
    Optimized version of simulate_trading_day that uses precomputed strategy decisions.
    """
    date_str = current_date.strftime("%Y-%m-%d")

    # calc previous day to retrieve action from precomuted decisions date to 1 day previous to avoid lookahead bias.
    previous_date = current_date - timedelta(days=1)
    date_str_prev = previous_date.strftime("%Y-%m-%d")  # lookahead

    logger.info(f"Simulating trading for {date_str}.")

    # new get daily regime data
    # daily_regime_data = ticker_price_history[regime_tickers[0]].loc[date_str]
    # current_vix_data = daily_regime_data["Close"].values[0]
    current_vix_data = ticker_price_history[regime_tickers[0]].at[date_str, "Close"]

    # df_daily_sp500 = ticker_price_history['^GSPC'].loc[date_str]
    # current_1day_sp500_return = df_daily_sp500['1day_return'].values[0]
    # current_1day_sp500_return = ticker_price_history['^GSPC'].loc[date_str].values[0]
    current_1day_sp500_return = ticker_price_history["^GSPC"].at[
        date_str, "1day_return"
    ]

    # print(f"{current_date} {current_vix_data = }, {current_1day_sp500_return}")
    current_regime_data = [current_vix_data, current_1day_sp500_return]

    for ticker in train_tickers:
        if date_str in ticker_price_history[ticker].index:
            daily_data = ticker_price_history[ticker].loc[date_str]
            current_price = daily_data["Close"]

            for strategy in strategies:
                strategy_name = strategy.__name__

                # Get precomputed strategy decision
                # action = precomputed_decisions[strategy_name][ticker].get(date_str)
                # # adjust start date to 1 day previous to avoid look-ahead bias.
                # # action = precomputed_decisions[strategy_name][ticker].get(
                # #     date_str_prev
                # # )  # lookahead

                # if action is None:
                #     # Skip if no precomputed decision (should not happen if properly precomputed)
                #     logger.warning(
                #         f"No precomputed decision for {ticker}, {strategy_name}, {date_str}"
                #     )
                #     continue

                # Get precomputed strategy decision from DataFrame
                action = precomputed_decisions[strategy_name][
                    (precomputed_decisions[strategy_name]["Strategy"] == strategy_name)
                    & (precomputed_decisions[strategy_name]["Ticker"] == ticker)
                    & (precomputed_decisions[strategy_name]["Date"] == date_str)
                ]["Action"].values

                if len(action) == 0:
                    # Skip if no precomputed decision (should not happen if properly precomputed)
                    logger.warning(
                        f"No precomputed decision for {ticker}, {strategy_name}, {date_str}"
                    )
                    continue

                action = action[0]

                # Get account details for trade size calculation
                account_cash = trading_simulator[strategy_name]["amount_cash"]
                portfolio_qty = (
                    trading_simulator[strategy_name]["holdings"]
                    .get(ticker, {})
                    .get("quantity", 0)
                )
                total_portfolio_value = trading_simulator[strategy_name][
                    "portfolio_value"
                ]

                # Compute trade decision and quantity based on precomputed action
                # if mode == "train":
                if mode == "train" or mode == "test":
                    decision, qty = compute_trade_quantities_only_buy_one(
                        action, portfolio_qty
                    )
                    # Execute trade
                    trading_simulator, points = execute_trade_only_buy_one(
                        decision,
                        qty,
                        ticker,
                        current_price,
                        strategy,
                        trading_simulator,
                        points,
                        time_delta,
                        portfolio_qty,
                        total_portfolio_value,
                        current_regime_data,
                        current_date,
                        logger,
                    )

                else:
                    decision, qty = compute_trade_quantities(
                        action,
                        current_price,
                        account_cash,
                        portfolio_qty,
                        total_portfolio_value,
                    )
                    # Execute trade
                    trading_simulator, points = execute_trade(
                        decision,
                        qty,
                        ticker,
                        current_price,
                        strategy,
                        trading_simulator,
                        points,
                        time_delta,
                        portfolio_qty,
                        total_portfolio_value,
                        current_regime_data,
                        current_date,
                        logger,
                    )

    return trading_simulator, points


def compute_trade_quantities(
    action, current_price, account_cash, portfolio_qty, total_portfolio_value
):
    """
    Computes trade decision and quantity based on the precomputed action.
    This replaces the quantity calculation part of simulate_strategy.
    """
    max_investment = total_portfolio_value * trade_asset_limit

    if action == "Buy":
        return "buy", min(
            int(max_investment // current_price), int(account_cash // current_price)
        )
    elif action == "Sell" and portfolio_qty > 0:
        return "sell", min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
    else:
        return "hold", 0


def compute_trade_quantities_only_buy_one(action, portfolio_qty):
    """
    Computes trade decision and quantity based on the precomputed action.
    This replaces the quantity calculation part of simulate_strategy.
    """
    if action == "Buy" and portfolio_qty == 0:
        return "buy", 1
    elif action == "Sell" and portfolio_qty > 0:
        return "sell", 1
    else:
        return "hold", 0


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


def precompute_strategy_decisions_json(
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

    # Initialize result structure
    precomputed_decisions = {
        strategy.__name__: {ticker: {} for ticker in train_tickers}
        for strategy in strategies
    }

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

    # Combine results from all processed days
    for day_results in results:
        if day_results:  # Skip empty results
            date_str = day_results["date"]
            for strategy_name, strategy_data in day_results["strategies"].items():
                for ticker, action in strategy_data.items():
                    precomputed_decisions[strategy_name][ticker][date_str] = action

    logger.info(
        f"Strategy decision precomputation complete. Processed {len(results)} trading days."
    )

    return precomputed_decisions


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
                historical_data = get_historical_data(
                    ticker, date, ideal_period[strategy_name], ticker_price_history
                )

                if historical_data is None or historical_data.empty:
                    continue

                # Compute strategy signal
                action = strategy(ticker, historical_data)
                result["strategies"][strategy_name][ticker] = action

            except Exception:
                # Skip errors in worker process
                continue

    return result


def prepare_regime_data(ticker_price_history, logger):
    """
    Prepares regime data for trading simulation by calculating and storing
    relevant metrics for the SP500 and VIX.

    This function calculates the 1-day percentage change in the SP500's closing price
    and stores it as a new column ('1day_return') in the SP500's DataFrame.
    It then saves the SP500 and VIX DataFrames to CSV files for later use.

    Args:
        ticker_price_history (dict): A dictionary where keys are ticker symbols
                                        and values are pandas DataFrames containing
                                        historical price data.
        logger (logging.Logger, optional): A logger object for logging messages.
                                            Defaults to the module's logger.

    Returns:
        dict: The updated ticker_price_history dictionary with the modified SP500 DataFrame.
    """
    try:
        df_sp500 = ticker_price_history.get("^GSPC")
        df_sp500["1day_return"] = df_sp500["Close"].pct_change()
        ticker_price_history["^GSPC"] = df_sp500
        save_df_to_csv(
            ticker_price_history["^GSPC"],
            "regime_sp500_data.csv",
            "results",
            logger=logger,
        )
        save_df_to_csv(
            ticker_price_history["^VIX"],
            "regime_vix_data.csv",
            "results",
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Error procesing regime data {e}")
    return ticker_price_history


if __name__ == "__main__":
    import logging
    from pymongo import MongoClient
    import certifi
    import sys

    # Ensure sys.path manipulation is at the top, before other local imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import FINANCIAL_PREP_API_KEY, mongo_url

    ca = certifi.where()

    # Set up logging
    logs_dir = "logs"
    # Create the directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(os.path.join(logs_dir, "train_test.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)

    period_start = "2019-01-01"
    period_end = "2021-01-01"
    train_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    # FINANCIAL_PREP_API_KEY,
    logger,

    ticker_price_history, ideal_period = initialize_simulation(
        period_start,
        period_end,
        train_tickers,
        mongo_client,
        FINANCIAL_PREP_API_KEY,
        logger,
    )

    print(ideal_period)
