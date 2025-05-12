import heapq
import logging
import logging.config
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from config import PRICE_DB_PATH
# from PriceData.store_trades_vectorised_v2 import prepare_sp500_one_day_return
from control import (
    benchmark_asset,
    minimum_cash_allocation,
    prediction_threshold,
    regime_tickers,
    test_period_end,
    test_period_start,
    train_start_cash,
    train_stop_loss,
    train_take_profit,
    train_tickers,
    train_trade_asset_limit,
    train_trade_liquidity_limit,
    train_trade_strategy_limit,
)
from helper_files.client_helper import (
    strategies,
    strategies_top10_acc,
)
from helper_files.train_client_helper import (
    calculate_metrics,
    generate_tear_sheet,
)
from log_config import LOG_CONFIG
from PriceData.store_rf_models import (
    check_model_exists,
    load_rf_model,
)
from random_forest import (
    predict_random_forest_classifier,
    train_random_forest_classifier_features,
)


def initialize_test_account(start_cash):
    """
    Initializes the test trading account with starting parameters

    account structure example:
        {
            "holdings": {
                "AAPL": {
                    "strategy1": {
                        "quantity": 10,
                        "price": 100,
                        "current_price": 150.0,
                        "current_value": 1500.0,
                    }
                },
                "MSFT": {
                    "strategy2": {
                        "quantity": 5,
                        "price": 200,
                        "current_price": 250.0,
                        "current_value": 1250.0,
                    }
                },
            },
            "holdings_value_by_strategy": {
                "strategy1": 1500.0,
                "strategy2": 1250.0,
            },
            "cash": 10000,
            "total_portfolio_value": 12750.0,
        }
    """
    return {
        "holdings": {},
        "holdings_value_by_strategy": {},
        "cash": start_cash,
        "fees": 0,
        "total_portfolio_value": start_cash,
    }


def check_stop_loss_take_profit_rtn_order(
    account, ticker, current_price, current_date, strategy_name
):
    """
    Checks and stop loss and take profit orders for a given ticker.

    Parameters:
    - account (dict): The current account state, including holdings, cash, and trades.
    - ticker (str): The ticker symbol of the asset to check.
    - current_price (float): The current price of the asset.

    Returns:
    - dict: The stop loss and take profit orders or None.
    """

    sell_order = None
    quantity = (
        account.get("holdings", {})
        .get(ticker, {})
        .get(strategy_name, {})
        .get("quantity", 0)
    )

    if quantity > 0:
        stop_loss = account["holdings"][ticker][strategy_name].get("stop_loss")
        take_profit = account["holdings"][ticker][strategy_name].get(
            "take_profit"
        )
        assert stop_loss is not None
        assert take_profit is not None
        if current_price < stop_loss:
            sell_order = {
                "strategy_name": strategy_name,
                "ticker": ticker,
                "action": "Sell",  # Should always be 'Sell' here
                "quantity": quantity,
                "current_price": current_price,
                "note": "Sell - stoploss",
                "date": current_date.strftime("%Y-%m-%d"),
            }
        elif current_price > take_profit:
            sell_order = {
                "strategy_name": strategy_name,
                "ticker": ticker,
                "action": "Sell",  # Should always be 'Sell' here
                "quantity": quantity,
                "current_price": current_price,
                "note": "Sell - take_profit",  # Should always be 'Sell' here
                "date": current_date.strftime("%Y-%m-%d"),
            }
    return sell_order


def execute_buy_orders(
    account,
    action,
    strategy_name,
    ticker,
    quantity,
    current_price,
    stop_loss_pct,
    take_profit_pct,
):
    """
    TODO: seperate updating account logic and combine with execute sell order account logic.
    Combine buy/sell account logic into 1 function for improved maintenance.

    TODO: update docstring.

    Executes buy orders from the buy and suggestion heaps.
    Creates stop-loss and take-profit prices.

    Parameters:
    - buy_heap (list): A heap of buy orders to execute.
    - suggestion_heap (list): A heap of suggested buy orders to execute.
    - account (dict): The current account state, including holdings, cash, and trades.
    - current_date (datetime): The current date for executing the buy orders.

    Returns:
    - dict: The updated account state after executing the buy orders.
    """
    if action == "Buy":
        # update account qty, sl, tp, and cash
        account["cash"] -= round(quantity * current_price, 2)
        account["cash"] = round(account["cash"], 2)

        if ticker not in account["holdings"]:
            account["holdings"][ticker] = {}

        if strategy_name not in account["holdings"][ticker]:
            account["holdings"][ticker][strategy_name] = {
                "quantity": 0,
                "price": 0,
                "stop_loss": 0,
                "take_profit": 0,
            }

        account["holdings"][ticker][strategy_name]["quantity"] += round(
            quantity, 2
        )
        account["holdings"][ticker][strategy_name]["price"] = current_price
        account["holdings"][ticker][strategy_name]["stop_loss"] = round(
            current_price * (1 - stop_loss_pct), 2
        )
        account["holdings"][ticker][strategy_name]["take_profit"] = round(
            current_price * (1 + take_profit_pct), 2
        )

        # broker fees
        account = update_account_fees_per_order(
            account, quantity, current_price
        )

    return account


def execute_sell_orders(
    action,
    ticker,
    strategy_name,
    account,
    current_price,
    current_date,
    note,
    logger,
):
    """
    Executes sell orders for a given ticker and strategy.

    Parameters:
    - action (str): The action to be executed, should be "sell".
    - ticker (str): The ticker symbol of the asset to be sold.
    - strategy_name (str): The name of the strategy under which the asset is held.
    - account (dict): The current account state, including holdings, cash, and trades.
    - current_price (float): The current price of the asset.
    - logger (Logger): The logger instance for logging information.

    Returns:
    - dict: The updated account state after executing the sell orders.
    """
    if (
        action == "Sell"
        and ticker in account["holdings"]
        and strategy_name in account["holdings"][ticker]
    ):
        # quantity = max(quantity, 1)
        quantity = account["holdings"][ticker][strategy_name]["quantity"]
        # trade_id_value = (
        #     f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}"
        # )

        # trade_df = pd.DataFrame(
        #     {
        #         # "trade_id": f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}",
        #         "date": current_date.strftime("%Y-%m-%d"),
        #         "symbol": ticker,
        #         "action": "sell",
        #         "quantity": quantity,
        #         "price": round(current_price, 2),
        #         "total_value": round(quantity * current_price, 2),
        #         "strategy": strategy_name,
        #         "note": note,
        #     },
        #     index=[trade_id_value],
        # )
        # trade_df.index.name = "trade_id"

        # insert_trade_into_trading_account_db(
        #     trade_df, trading_account_db_name, experiment_name
        # )

        account["cash"] += quantity * current_price
        account["cash"] = round(account["cash"], 2)

        del account["holdings"][ticker][strategy_name]
        if account["holdings"][ticker] == {}:
            del account["holdings"][ticker]

        # broker fees
        account = update_account_fees_per_order(
            account, quantity, current_price
        )

        logger.info(
            f"{ticker} - Sold {quantity} shares at ${current_price} for {strategy_name} date: {current_date.strftime('%Y-%m-%d')}"
        )
    return account


def update_account_fees_per_order(account, quantity, current_price):
    # fee = broker_commission_pct * quantity * current_price
    fee = calculate_ibkr_us_stock_fee(
        quantity, current_price, pricing_type="fixed"
    )
    account["cash"] -= fee
    account["cash"] = round(account["cash"], 2)
    account["fees"] = account["fees"] + fee
    account["fees"] = round(account["fees"], 2)
    return account


def calculate_ibkr_us_stock_fee(
    quantity, price_per_share, pricing_type="fixed"
):
    """
    Calculate the trading fee for US stocks on Interactive Brokers (IBKR), including minimum fees.

    Parameters:
    - quantity (int): Number of shares being traded.
    - price_per_share (float): Price of each share.
    - pricing_type (str): "fixed" or "tiered" pricing model.

    Returns:
    - float: Trading fee for the transaction.

    # Example usage:
    print(calculate_ibkr_us_stock_fee(50, 2, "fixed"))  # Expected output: 1.00 (minimum fee applied)
    print(calculate_ibkr_us_stock_fee(500, 10, "fixed"))  # Expected output: 2.50
    print(calculate_ibkr_us_stock_fee(50, 2, "tiered"))  # Expected output: 0.35 (minimum fee applied)
    print(calculate_ibkr_us_stock_fee(500, 10, "tiered")) # Expected output: 1.75
    """
    trade_value = quantity * price_per_share

    if pricing_type == "fixed":
        fee = min(
            max(quantity * 0.005, 1), trade_value * 0.01
        )  # Fixed: $0.005 per share, minimum $1, capped at 1% of trade
    elif pricing_type == "tiered":
        # Assuming lowest tier pricing for simplicity
        fee = max(
            quantity * 0.0035, 0.35
        )  # Tiered: $0.0005 to $0.0035 per share, minimum $0.35 per trade
    else:
        raise ValueError("Invalid pricing_type. Choose 'fixed' or 'tiered'.")

    return round(fee, 2)


# not needed? incorporated into update_account_portfolio_values
def get_holdings_value_by_strategy(
    account, ticker_price_history, current_date
):
    """
    Calculates the current value of existing holdings by strategy.

    Parameters:
    - account (dict): The current account state, including holdings, cash, and total portfolio value.
    - ticker_price_history (dict): Dictionary containing historical price data for tickers.
    - current_date (datetime): The current date for which to calculate the holdings value.

    Returns:
    - dict: Dictionary with the current value of holdings by strategy.
    """
    holdings_value_by_strategy = {}
    for ticker, strategies in account["holdings"].items():
        for strategy, holding in strategies.items():
            current_price = ticker_price_history[ticker].loc[
                current_date.strftime("%Y-%m-%d")
            ]["Close"]
            holding_value = holding["quantity"] * current_price

            if strategy not in holdings_value_by_strategy:
                holdings_value_by_strategy[strategy] = 0
            holdings_value_by_strategy[strategy] += holding_value

    return holdings_value_by_strategy


def strategy_and_ticker_cash_allocation(
    buy_orders_df,
    account,
    prediction_threshold,
    asset_limit,
    strategy_limit,
    trade_liquidity_limit_cash,
    minimum_cash_allocation,
    logger,
):
    """
    Allocates cash to strategies and tickers based on prediction results and constraints.

    Parameters:
    - buy_orders_df (pd.DataFrame): DataFrame containing prediction results, including strategy names, tickers, actions, predictions, probabilities and accuracies.
    - account (dict): The current account state, including holdings, cash, and total portfolio value.
    - prediction_threshold (float): The minimum score required for a strategy to qualify.
    - asset_limit (float): The maximum proportion of the portfolio that can be allocated to a single asset.
    - strategy_limit (float): The maximum proportion of the portfolio that can be allocated to a single strategy.
    - trade_liquidity_limit_cash (float): how much spare cash to keep.

    Returns:
    - pd.DataFrame: DataFrame containing the allocated cash for each strategy and ticker.
    """
    if buy_orders_df.empty:
        logger.info("buy_orders_df is empty.")
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = ["probability", "prediction", "accuracy", "action"]
    if not all(col in buy_orders_df.columns for col in required_cols):
        logger.error(
            f"Missing required columns in buy_orders_df. Found: {buy_orders_df.columns}"
        )
        return pd.DataFrame()

    # Filter for actual buy signals confirmed by RF (prediction == 1)
    # Note: The loop appending to prediction_results_list already ensures action=='Buy' and prediction==1
    confirmed_buys_df = buy_orders_df[buy_orders_df["prediction"] == 1].copy()

    if confirmed_buys_df.empty:
        logger.info("No confirmed buy signals (prediction == 1).")
        return pd.DataFrame()

    # Use probability as the primary score for ranking
    confirmed_buys_df["score"] = confirmed_buys_df["probability"]

    # Sort by the new probability-based score
    confirmed_buys_df = confirmed_buys_df.sort_values(
        by=["score"], ascending=False
    )
    # logger.info(
    #     f"Sorted confirmed buys by probability score:
    # \n{confirmed_buys_df[['strategy_name', 'ticker', 'probability', 'score']]}"
    # )

    # Filter DataFrame where score (probability) > prediction_threshold
    qualifying_strategies_df = confirmed_buys_df[
        confirmed_buys_df["score"] >= prediction_threshold
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    if qualifying_strategies_df.empty:
        logger.info(
            f"No strategies qualify after probability threshold {prediction_threshold}."
        )
        return pd.DataFrame()

    # logger.info(
    #     f"Qualifying strategies after threshold:\n{qualifying_strategies_df[['strategy_name', 'ticker', 'score']]}"
    # )

    total_portfolio_value = account["total_portfolio_value"]
    # available_cash = account["cash"]
    available_cash = account["cash"] - trade_liquidity_limit_cash

    max_strategy_investment = total_portfolio_value * strategy_limit
    max_strategy_investment = round(max_strategy_investment, 2)

    logger.info(f"{max_strategy_investment = }")

    qualifying_strategies_df["max_s_cash"] = max_strategy_investment

    # Adjust strategy allocation based on existing holdings
    holdings_value_df = pd.DataFrame.from_dict(
        # holdings_value_by_strategy, orient="index", columns=["holdings_value"]
        account["holdings_value_by_strategy"],
        orient="index",
        columns=["holdings_value"],
    )

    qualifying_strategies_df = qualifying_strategies_df.merge(
        holdings_value_df,
        left_on="strategy_name",
        right_index=True,
        how="left",
    )

    qualifying_strategies_df["holdings_value"] = (
        qualifying_strategies_df["holdings_value"]
        .fillna(0)
        .infer_objects(copy=False)
    )

    # note: clip to avoid negative
    qualifying_strategies_df["max_s_cash_adj"] = (
        qualifying_strategies_df["max_s_cash"]
        - qualifying_strategies_df["holdings_value"]
    ).clip(lower=0)

    # Adjust by number_of_tickers_in_qualifying_strategies
    qualifying_strategies_df["ticker_count"] = (
        qualifying_strategies_df.groupby("strategy_name")["ticker"].transform(
            "count"
        )
    )

    # Max available cash for each strategy divided by the number of tickers in that strategy
    qualifying_strategies_df["max_s_t_cash"] = (
        qualifying_strategies_df["max_s_cash_adj"]
        / qualifying_strategies_df["ticker_count"]
    )

    # TODO: if this value is too low we dont want to make order.
    # Calculate probability adjusted investment (using 'score' which is now probability)
    qualifying_strategies_df["prob_adj_investment"] = (
        qualifying_strategies_df["max_s_t_cash"]
        * qualifying_strategies_df[
            "score"
        ]  # Use score (probability) for weighting
    )

    # Calculate investment considering asset limit
    asset_limit_value = asset_limit * total_portfolio_value
    asset_limit_value = round(asset_limit_value, 2)
    logger.info(f"{asset_limit_value=}")

    # Allocate cash based on prob_adj_investment until cash is exhausted
    qualifying_strategies_df["allocated_cash"] = (
        0.0  # Ensure the column is of float type
    )

    for ticker, group in qualifying_strategies_df.groupby("ticker"):
        # Calculate the remaining cash available for this ticker after considering existing holdings
        existing_holding_value = sum(
            holding["quantity"] * group.iloc[0]["current_price"]
            for strategy, holding in account["holdings"]
            .get(ticker, {})
            .items()
        )
        remaining_cash = min(
            max(0, asset_limit_value - existing_holding_value), available_cash
        )

        for index, row in group.iterrows():
            if remaining_cash <= 0:
                # if remaining_cash <= trade_liquidity_limit_cash:
                break
            # Allocate based on probability-adjusted investment suggestion
            allocation = min(row["prob_adj_investment"], remaining_cash)
            qualifying_strategies_df.at[index, "allocated_cash"] = round(
                (allocation), 2
            )
            remaining_cash -= allocation
            available_cash -= allocation

    qualifying_strategies_df["quantity"] = (
        qualifying_strategies_df["allocated_cash"]
        / qualifying_strategies_df["current_price"]
    ).round(2)

    logger.info(
        f"total allocated cash = {qualifying_strategies_df['allocated_cash'].sum()}"
    )

    # minium_cash_allocation = 100  # cash
    return qualifying_strategies_df[
        [
            "strategy_name",
            "ticker",
            "action",
            "current_price",
            "allocated_cash",
            "quantity",
            "date",
            "prediction",
            "probability",
            "accuracy",
            "score",
        ]
    ][qualifying_strategies_df["allocated_cash"] > minimum_cash_allocation]


# not needed?
def create_buy_heap(buy_df):
    """
    Creates a heap of buy orders from the given DataFrame.

    Parameters:
    - buy_df (pd.DataFrame): DataFrame containing buy orders with columns 'score', 'quantity', 'ticker', 'current_price', and 'strategy_name'.

    Returns:
    - list: A heap of buy orders sorted by score.
    """
    buy_heap = []
    # Ensure the DataFrame is not empty
    if buy_df.empty:
        return buy_heap

    for index, row in buy_df.iterrows():
        heapq.heappush(
            buy_heap,
            (
                -row["score"],
                row["quantity"],
                row["ticker"],
                row["current_price"],
                row["strategy_name"],
            ),
        )
    return buy_heap


def update_account_portfolio_values(
    account, ticker_price_history, current_date
):
    """
    Updates the account dictionary with the latest portfolio values based on current prices.

    Updates each holding with the latest current_price and recalculates its current_value.
    Aggregates the value of holdings by strategy in holdings_value_by_strategy.
    Computes the total portfolio value as the sum of cash and all holdings.

    Args:
        account (dict): The account data containing 'cash', 'holdings', and other info.
        ticker_price_history (dict): Mapping of ticker symbols to their historical price DataFrames.
        current_date (datetime): The date for which to fetch the latest closing prices.

    Returns:
        dict: The updated account dictionary with:
            - 'holdings_value_by_strategy': Value of holdings grouped by strategy.
            - 'total_portfolio_value': Total value of the portfolio (cash + holdings).
            - Updates to each holding with 'current_price' and 'current_value'.

    Notes:
        - Assumes each holding contains a 'quantity' field.
        - Expects price DataFrames to have a 'Close' column and be indexed by date strings.
    """

    # Calculate and update total portfolio value
    total_value = account["cash"]
    account["holdings_value_by_strategy"] = {}
    for ticker, account_strategies in account["holdings"].items():
        for strategy, holding in account_strategies.items():
            try:
                current_price = float(
                    ticker_price_history[ticker].loc[
                        current_date.strftime("%Y-%m-%d")
                    ]["Close"]
                )
            except (KeyError, ValueError):
                # Handle missing price data gracefully
                current_price = holding.get("current_price", 0)

            holding["current_price"] = current_price
            # value of each holding in each strategy
            current_value = round(holding["quantity"] * current_price, 2)
            holding["current_value"] = current_value

            # value of holdings by strategy
            if strategy not in account["holdings_value_by_strategy"]:
                account["holdings_value_by_strategy"][strategy] = 0
            account["holdings_value_by_strategy"][strategy] += current_value
            account["holdings_value_by_strategy"][strategy] = round(
                account["holdings_value_by_strategy"][strategy], 2
            )

            # total value of portfolio
            total_value += current_value
    account["total_portfolio_value"] = round(total_value, 2)

    return account


def insert_trade_into_trading_account_db(
    trades_df, trading_account_db_name, experiment_name
):
    """
    Inserts trades into the trading account database.

    Parameters:
    - trades_df (pd.DataFrame): DataFrame containing trade data.
    - trading_account_db_name (str): path to the trading account database.
    - experiment_name (str): Name of the experiment for which the trades are being inserted.

    Returns:
    - None
    """
    # Ensure the DataFrame is not empty
    if trades_df.empty:
        logger.warning("No trades to insert into the database.")
        return

    # order_id check if column name exists
    if "order_id" not in trades_df.columns:
        # Add a new column with the order_id value
        trades_df["order_id"] = (
            trades_df["ticker"]
            + "_"
            + trades_df["strategy_name"]
            + "_"
            + trades_df["date"].astype(str)
        )
        trades_df.set_index("order_id", inplace=True)

    try:
        with sqlite3.connect(trading_account_db_name) as conn:
            trades_df.to_sql(
                f"trades_{experiment_name}",
                conn,
                if_exists="append",
                index=True,
                dtype={"order_id": "TEXT PRIMARY KEY"},
            )
    except Exception as e:
        logger.error(f"Error saving trades to database: {trades_df=} {e}")


def summarize_account_tickers_by_value(account):
    """
    Returns a dict mapping each ticker to its total current value in the account,
    using the 'current_price' stored in the account dict.
    Prints a summary to stdout.
    """
    ticker_values = {}
    holdings = account.get("holdings", {})
    for ticker, strategies in holdings.items():
        total_value = 0
        for strategy_name, pos in strategies.items():
            value = pos.get("current_value", 0)
            total_value += value
        ticker_values[ticker] = round(total_value, 2)
    return ticker_values


def process_orders(
    orders_df,
    account,
    date,
    trade_liquidity_limit_cash,
    stop_loss_pct,
    take_profit_pct,
    logger,
):
    """
    Process the interval (daily) orders list.
    orders list contains sell and buy orders.
    First process sell orders, then assign allocation for buy orders.
    buy orders should be processed by probability (score).
    """
    orders_df["executed"] = 0
    sell_orders_df = orders_df[orders_df["action"] == "Sell"]
    buy_orders_df = orders_df[orders_df["action"] == "Buy"]
    num_sell_orders = len(sell_orders_df)
    num_buy_orders = len(buy_orders_df)

    logger.info(
        f"Pre allocation: {len(orders_df)=}, {num_sell_orders=}, {num_buy_orders=}"
    )

    if num_sell_orders > 0:
        logger.info("process sell orders...")
        # processed_sell_orders = []
        # failed_sell_orders = []

        for index, row in sell_orders_df.iterrows():
            try:
                account = execute_sell_orders(
                    row["action"],
                    row["ticker"],
                    row["strategy_name"],
                    account,
                    row["current_price"],
                    date,
                    row["note"],
                    logger,
                )
                sell_orders_df.at[index, "executed"] = 1
            except Exception as e:
                logger.error(f"error executing sell order {e}")

        # logger.info(f"{sell_orders_df=}")
        failed_sell_orders_df = sell_orders_df[sell_orders_df["executed"] == 0]

        if len(failed_sell_orders_df) > 0:
            logger.error(f"Error failed sell orders: {failed_sell_orders_df=}")

    # Execute buy orders, create sl and tp prices
    buy_df = pd.DataFrame()  # Initialize buy_df as an empty DataFrame
    if num_buy_orders > 0:
        logger.info("process buy orders...")
        # update buy_order with allocated_cash and qty
        buy_df = strategy_and_ticker_cash_allocation(
            buy_orders_df,
            account,
            prediction_threshold,
            train_trade_asset_limit,
            train_trade_strategy_limit,
            trade_liquidity_limit_cash,
            minimum_cash_allocation,
            logger,
        )
        logger.info(
            f"Post allocation: {len(orders_df)=}, {num_sell_orders=}, {len(buy_df)=}"
        )
        # logger.info(f"debug\n\n{buy_df}")
        buy_df["executed"] = 0
        for index, row in buy_df.iterrows():
            try:
                account = execute_buy_orders(
                    account,
                    row["action"],
                    row["strategy_name"],
                    row["ticker"],
                    row["quantity"],
                    row["current_price"],
                    stop_loss_pct,
                    take_profit_pct,
                )
                buy_df.at[index, "executed"] = 1
            except Exception as e:
                logger.error(f"error executing buy order {e}")

        # logger.info(f"{buy_df=}")
        failed_buy_df = buy_df[buy_df["executed"] == 0]

        if len(failed_buy_df) > 0:
            logger.error(f"Error failed sell orders: {failed_buy_df=}")

    # concatenate buy and sell orders into one dataframe
    orders_df = pd.concat([sell_orders_df, buy_df], ignore_index=True)

    return account, orders_df


def main_test_loop(
    account,
    test_date_range,
    tickers_list,
    ticker_price_history,
    precomputed_decisions,
    use_rf_model_predictions,
    train_rf_classifier,
    account_values,
    trading_account_db_name,
    rf_dict,
    experiment_name,
    train_trade_liquidity_limit,
    logger,
):
    """
    Design desicions:
        - If have position (qty>0): skip subsequent buy orders?
        (otherwise fills up any remaining allocation with small orders).

        - Minimum order size?

        - combine execute buy and sell into one execute orders function?
        pros: simpler logic.
    """
    for date in test_date_range:
        logger.info(
            f"\n\n\n------------- NEW INTERVAL {date.strftime("%Y-%m-%d")}"
        )
        orders_list = []  # combine buy and sell orders into one list
        date_missing = False
        # logger.info(f"{account=}")
        for idx, strategy in enumerate(strategies):
            strategy_name = strategy.__name__
            logger.info(
                f"{strategy_name} {idx + 1}/{len(strategies)} {date.strftime("%Y-%m-%d")}"
            )

            # check if current date is in precomputed_decisions
            if (
                date.strftime("%Y-%m-%d")
                not in precomputed_decisions[strategy_name].index
            ):
                logger.warning(
                    f"Date {date.strftime('%Y-%m-%d')} not found in precomputed decisions for {strategy_name}."
                )
                date_missing = True
                continue

            # retrain model with newer data
            if train_rf_classifier:
                # Load trades data (training data)
                # Get one day before
                previous_day = date - timedelta(days=1)
                historic_trades_df = get_trades_training_data_from_db(
                    strategy_name,
                    trades_list_db_name,
                    logger,
                    None,
                    previous_day.strftime("%Y-%m-%d"),
                )
                logger.info(f"{len(historic_trades_df)=}")
                logger.info(f"{historic_trades_df=}")

                # required_features = []
                # train_random_forest_classifier_features(
                #     historic_trades_df, required_features
                # )

            for ticker in tickers_list:
                try:
                    # Attempt to get the shifted decision
                    action = (
                        precomputed_decisions[strategy_name]
                        .shift(1)
                        .loc[date.strftime("%Y-%m-%d"), ticker]
                    )
                    # logger.info(f"got action: {action}")
                except KeyError:
                    # Log a warning and default action if the key is not found
                    logger.warning(
                        f"Precomputed decision not found for {ticker} on {date.strftime('%Y-%m-%d')} after shift. Defaulting to 'Hold'."
                    )
                    action = "Hold"

                try:
                    current_price = ticker_price_history[ticker].loc[
                        date.strftime("%Y-%m-%d")
                    ]["Close"]
                    current_price = float(round(current_price, 2))
                    # logger.info(
                    #     f"Current price: {ticker} {current_price} on {date.strftime('%Y-%m-%d')}."
                    # )
                except KeyError:
                    logger.warning(
                        f"Current price not found for {ticker} on {date.strftime('%Y-%m-%d')}."
                    )
                    continue

                # get any current holdings. needed for sl/tp and sell qty.
                quantity = (
                    account.get("holdings", {})
                    .get(ticker, {})
                    .get(strategy_name, {})
                    .get("quantity", 0)
                )

                if quantity > 0:
                    sl_tp_sell_order = check_stop_loss_take_profit_rtn_order(
                        account,
                        ticker,
                        current_price,
                        date,
                        strategy_name,
                    )
                    if sl_tp_sell_order:
                        # append sell order
                        logger.info(f"{sl_tp_sell_order=}")
                        # sell_daily_list.append(sl_tp_sell_order)
                        orders_list.append(sl_tp_sell_order)
                        continue

                # append sell orders to daily list.
                if action == "Sell" and quantity > 0:
                    # sell_daily_list.append(
                    orders_list.append(
                        {
                            "strategy_name": strategy_name,
                            "ticker": ticker,
                            "action": action,  # Should always be 'Sell' here
                            "quantity": quantity,
                            "current_price": current_price,
                            "note": "",
                            "date": date.strftime("%Y-%m-%d"),
                        }
                    )
                    continue

                # logger.info(f"{account = }")

                elif action == "Buy" and quantity == 0:
                    # used if use_rf_model_predictions == False. ie baseline test.
                    prediction = 1
                    accuracy = 1
                    probability = 1

                    # only load rf_model if buy signal
                    if use_rf_model_predictions:
                        # Check if model is ALREADY LOADED in memory (in rf_dict)
                        if strategy_name in rf_dict:
                            logger.info(
                                f"Model for {strategy_name} already loaded in memory."
                            )
                            if rf_dict[strategy_name] is None:
                                logger.error(
                                    f"Model for {strategy_name} is None, skipping..."
                                )
                                continue
                        else:
                            if check_model_exists(strategy_name):
                                rf_dict[strategy_name] = load_rf_model(
                                    strategy_name, logger
                                )
                                if rf_dict[strategy_name] is None:
                                    logger.error(
                                        f"Model for {strategy_name} could not be loaded, skipping..."
                                    )
                                    continue
                            else:
                                logger.error(
                                    f"Model for {strategy_name} does not exist, skipping..."
                                )
                                continue

                        assert isinstance(
                            rf_dict[strategy_name], dict
                        ), "loaded_model is not a dictionary, model loading failed."
                        # logger.info(f"{rf_dict}")

                        # prepare regime data for prediction
                        daily_vix_df = ticker_price_history["^VIX"].loc[
                            date.strftime("%Y-%m-%d")
                        ]["Close"]
                        assert daily_vix_df is not None, "daily_vix_df is None"
                        One_day_spy_return = ticker_price_history["^GSPC"].loc[
                            date.strftime("%Y-%m-%d")
                        ]["One_day_spy_return"]
                        assert (
                            One_day_spy_return is not None
                        ), "One_day_spy_return is None"

                        data = {
                            "^VIX": [daily_vix_df],
                            "One_day_spy_return": [One_day_spy_return],
                        }
                        sample_df = pd.DataFrame(data, index=[0])

                        # Get prediction (0 or 1) and probability of class 1 (positive return)
                        prediction, probability, probabilities = (
                            predict_random_forest_classifier(
                                rf_dict[strategy_name]["rf_classifier"],
                                sample_df,
                            )
                        )
                        logger.info(f"{probabilities=}")
                        if prediction != 1:
                            action = "Hold"  # Override original 'Buy' if RF doesn't confirm

                        accuracy = round(
                            rf_dict[strategy_name]["accuracy"], 2
                        )  # Keep accuracy for logging/potential future use
                        probability = np.round(
                            probability[0], 4
                        )  # Round probability
                        logger.info(f"{probability = }")

                        logger.info(
                            f"Prediction {date.strftime('%Y-%m-%d')} {strategy_name} {ticker}: {prediction} (Prob: {probability:.4f}), Acc: {accuracy}, VIX: {daily_vix_df:.2f}, SPY: {One_day_spy_return:.2f}, Action: {action}"
                        )

                    # Only add to results if the original action was Buy and RF prediction is 1
                    if action == "Buy":

                        orders_list.append(
                            {
                                "strategy_name": strategy_name,
                                "ticker": ticker,
                                "action": action,  # Should always be 'Buy' here
                                "prediction": prediction,  # Should always be 1 here
                                "accuracy": accuracy,  # Historical accuracy of the model
                                "probability": probability,  # Probability of this specific prediction being 1
                                "current_price": current_price,
                                "date": date.strftime("%Y-%m-%d"),
                            }
                        )

                elif action == "Buy" and quantity > 0:
                    # logger.info(
                    #     f"Not appending new order for Buy signal because already have holding. {action=} {quantity=} {ticker}"
                    # )
                    continue
                elif action == "Sell" and quantity == 0:
                    continue
                elif action == "Hold":
                    continue
                else:
                    logger.error(
                        f"unexpected action, or action/qty combination {action=} {quantity=}"
                    )

        # end of each date, merge buy and sell orders, then execute.
        if not date_missing:
            # TODO: merge orders for same ticker (but still update holdings by strategy/ticker)
            logger.info(
                f"\n\n++ EXECUTE DAILY ORDERS {date.strftime("%Y-%m-%d")}"
            )
            # Calculate and update account total portfolio value
            account = update_account_portfolio_values(
                account, ticker_price_history, date
            )

            logger.debug(
                f"before processing orders: {account["holdings_value_by_strategy"]=}"
            )

            # daily orders summary
            # convert orders_list into a df
            orders_df = pd.DataFrame(orders_list)

            # Required columns
            required_cols = [
                "strategy_name",
                "ticker",
                "action",
                "note",
                "current_price",
                "allocated_cash",
                "quantity",
                "prediction",
                "probability",
                "accuracy",
                "score",
                "date",
                "executed",
            ]

            # Find missing columns
            missing_cols = [
                col for col in required_cols if col not in orders_df.columns
            ]

            # Add missing columns with default NaN values
            for col in missing_cols:
                orders_df[col] = np.nan

            if not orders_df.empty:
                # summarize orders: use orders_df to show number of buy and sell orders by strategy
                orders_summary = orders_df.groupby(
                    ["strategy_name", "action"]
                ).size()
                logger.info(f"Orders summary:\n{orders_summary}")

                # execute orders and update account
                account, orders_df = process_orders(
                    orders_df,
                    account,
                    date,
                    train_trade_liquidity_limit,
                    train_stop_loss,
                    train_take_profit,
                    logger,
                )

                # Calculate and update account total portfolio value
                account = update_account_portfolio_values(
                    account, ticker_price_history, date
                )
                logger.info(f"orders_df:\n\n{orders_df}")

                insert_trade_into_trading_account_db(
                    orders_df, trading_account_db_name, experiment_name
                )
            else:
                logger.info("no orders to process")

            # Update account values for metrics
            total_value = account["total_portfolio_value"]
            account_values[date] = total_value
            # logger.info(f"{total_value = }")
            logger.info(f"total_portfolio_value: {round(total_value, 2)}")
            """
            Daily summary of account.
            possibly change since last account.
            """
            # num_of_holdings = len(account["holdings"])
            # logger.info(account)
            logger.info(f"cash: {account['cash']}")
            logger.info(f"fees: {account['fees']}")
            logger.info(f"holdings_value_by_strategy:")
            logger.info(account["holdings_value_by_strategy"])
            ticker_values = summarize_account_tickers_by_value(account)
            logger.info(f"num tickers: {len(account["holdings"])}")
            logger.info(ticker_values)

    # Log final results
    # Convert account_values (Series) to DataFrame with Date as index and a column for portfolio value
    account_values_df = account_values.dropna().to_frame(
        name="total_portfolio_value"
    )
    account_values_df.index.name = "Date"
    # logger.info(f"{account_values_df=}")
    insert_account_values_into_db(
        account_values_df, trading_account_db_name, experiment_name
    )

    logger.info("-------------------------------------------------")
    # logger.info(f"Trades: {len(account['trades'])}")
    logger.info(f"Holdings: {account['holdings']}")
    logger.info(f"Account Cash: ${account['cash']: ,.2f}")
    logger.info(f"Fees paid: ${account['fees']}")
    logger.info(
        f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}"
    )
    logger.info(f"Account Values: {account_values}")
    # logger.info(f"Active Count: {active_count}")
    logger.info("-------------------------------------------------")

    try:
        # Calculate final metrics and generate tear sheet
        metrics = calculate_metrics(account_values)
        logger.info(metrics)
        generate_tear_sheet(
            account_values,
            filename=f"{benchmark_asset}_vs_strategy_{experiment_name}",
        )
        logger.info("Tear sheet generated.")
    except Exception as e:
        logger.error(f"Error generating tear sheet: {e}")
    return account


def insert_account_values_into_db(
    account_values_df, trading_account_db_name, experiment_name
):
    """
    Inserts account_values into the trading account database.

    Parameters:
    - account_values_df (pd.DataFrame): DataFrame containing trade data.
    - trading_account_db_name (str): path to the trading account database.
    - experiment_name (str): Name of the experiment for which the account_values are being inserted.

    Returns:
    - None
    """
    # Ensure the DataFrame is not empty
    if account_values_df.empty:
        logger.warning("No account_values to insert into the database.")
        return
    try:
        with sqlite3.connect(trading_account_db_name) as conn:
            account_values_df.to_sql(
                f"account_values_{experiment_name}",
                conn,
                if_exists="replace",
                index=True,
                dtype={"Date": "TEXT PRIMARY KEY"},
            )
    except Exception as e:
        logger.error(f"Error saving account_values to database: {e}")


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
    # import pandas as pd

    # Read S&P 500 data from the database
    query = "SELECT * FROM '^GSPC'"
    sp500_df = pd.read_sql_query(query, conn)

    # Ensure Date column is in datetime format
    if "Date" in sp500_df.columns:
        sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])

    # sp500_df.drop(columns=["1_day_spy_return"], inplace=True)
    # Calculate one-day percentage return using vectorized operations
    sp500_df["One_day_spy_return"] = (
        sp500_df["Close"].pct_change().round(4) * 100
    )
    sp500_df["One_day_spy_return"] = sp500_df["One_day_spy_return"].round(2)

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


def get_trades_training_data_from_db(
    strategy_name, trades_list_db_name, logger, start_date=None, end_date=None
):
    trades_df = pd.DataFrame()
    query = f"SELECT * FROM {strategy_name}"
    params = []
    if start_date or end_date:
        query += " WHERE"
        if start_date:
            query += " buy_date >= ?"
            params.append(start_date)
        if end_date:
            if start_date:
                query += " AND"
            query += " sell_date <= ?"
            params.append(end_date)
    with sqlite3.connect(trades_list_db_name) as trades_conn:
        try:
            trades_df = pd.read_sql(query, trades_conn, params=params)
        except Exception as e:
            logger.error(
                f"Error loading trades data for {strategy_name}, skipping: {e}"
            )
            return trades_df
    # trades_df["returnB"] = np.where(trades_df["ratio"] > 1, 1, 0)
    trades_df["return"] = np.where(trades_df["ratio"] > 1, 1, 0)
    # trades_df["baseline_pred"] = 1
    trades_df["buy_date"] = pd.to_datetime(
        trades_df["buy_date"]
    ).dt.normalize()
    trades_df["sell_date"] = pd.to_datetime(
        trades_df["sell_date"]
    ).dt.normalize()
    return trades_df


# examples to help in refactoring holdings dict to holdings_df
# ai
def update_holdings_from_orders(holdings_df, orders_df):
    # Process executed buys
    executed_buys = orders_df[
        (orders_df["action"] == "Buy") & (orders_df["executed"] == 1)
    ]
    if executed_buys.empty:
        return holdings_df

    executed_buys["total_cost"] = (
        executed_buys["quantity"] * executed_buys["current_price"]
    )
    new_holdings = (
        executed_buys.groupby(["ticker", "strategy_name"])
        .agg(
            quantity=("quantity", "sum"),
            total_cost=("total_cost", "sum"),
            last_price=("current_price", "last"),
        )
        .reset_index()
    )
    new_holdings["avg_price"] = (
        new_holdings["total_cost"] / new_holdings["quantity"]
    )

    # Merge with existing holdings
    if holdings_df is not None and not holdings_df.empty:
        holdings_df = pd.concat([holdings_df, new_holdings])
        holdings_df = (
            holdings_df.groupby(["ticker", "strategy_name"])
            .agg(
                quantity=("quantity", "sum"),
                total_cost=("total_cost", "sum"),
                last_price=("last_price", "last"),
            )
            .reset_index()
        )
        holdings_df["avg_price"] = (
            holdings_df["total_cost"] / holdings_df["quantity"]
        )
    else:
        holdings_df = new_holdings

    return holdings_df


# ai
def update_holdings_from_orders_with_sells(holdings_df, orders_df):
    # Process executed buys
    executed_buys = orders_df[
        (orders_df["action"] == "Buy") & (orders_df["executed"] == 1)
    ]
    if not executed_buys.empty:
        executed_buys["total_cost"] = (
            executed_buys["quantity"] * executed_buys["current_price"]
        )
        new_holdings = (
            executed_buys.groupby(["ticker", "strategy_name"])
            .agg(
                quantity=("quantity", "sum"),
                total_cost=("total_cost", "sum"),
                last_price=("current_price", "last"),
            )
            .reset_index()
        )
        new_holdings["avg_price"] = (
            new_holdings["total_cost"] / new_holdings["quantity"]
        )

        # Merge with existing holdings
        if holdings_df is not None and not holdings_df.empty:
            holdings_df = pd.concat([holdings_df, new_holdings])
            holdings_df = (
                holdings_df.groupby(["ticker", "strategy_name"])
                .agg(
                    quantity=("quantity", "sum"),
                    total_cost=("total_cost", "sum"),
                    last_price=("last_price", "last"),
                )
                .reset_index()
            )
            holdings_df["avg_price"] = (
                holdings_df["total_cost"] / holdings_df["quantity"]
            )
        else:
            holdings_df = new_holdings

    # Process executed sells
    executed_sells = orders_df[
        (orders_df["action"] == "Sell") & (orders_df["executed"] == 1)
    ]
    if not executed_sells.empty:
        sell_agg = (
            executed_sells.groupby(["ticker", "strategy_name"])
            .agg(quantity=("quantity", "sum"))
            .reset_index()
        )

        # Update holdings by reducing the quantity sold
        if holdings_df is not None and not holdings_df.empty:
            holdings_df = holdings_df.merge(
                sell_agg, on=["ticker", "strategy_name"], how="left"
            )
            holdings_df["quantity"] = holdings_df["quantity"] - holdings_df[
                "quantity_y"
            ].fillna(0)
            holdings_df.drop(columns=["quantity_y"], inplace=True)

            # Remove positions with zero or negative quantity
            holdings_df = holdings_df[holdings_df["quantity"] > 0]

            # Recalculate total cost & avg price
            holdings_df["total_cost"] = (
                holdings_df["avg_price"] * holdings_df["quantity"]
            )
        else:
            holdings_df = holdings_df  # No impact if there are no holdings

    return holdings_df


# ai
def update_holdings_from_orders_with_sells_v2(holdings_df, orders_df):
    # Process executed buys efficiently
    executed_buys = orders_df.loc[
        (orders_df["action"] == "Buy") & (orders_df["executed"] == 1)
    ]
    if not executed_buys.empty:
        executed_buys["total_cost"] = (
            executed_buys["quantity"] * executed_buys["current_price"]
        )

        # Use vectorized grouping with NumPy operations
        new_holdings = executed_buys.groupby(
            ["ticker", "strategy_name"], as_index=False
        ).agg(
            quantity=("quantity", "sum"),
            total_cost=("total_cost", "sum"),
            last_price=("current_price", "last"),
        )
        new_holdings["avg_price"] = (
            new_holdings["total_cost"] / new_holdings["quantity"]
        )

        # Merge with existing holdings
        if holdings_df is not None and not holdings_df.empty:
            holdings_df = pd.concat(
                [holdings_df, new_holdings], ignore_index=True
            )
            holdings_df = holdings_df.groupby(
                ["ticker", "strategy_name"], as_index=False
            ).agg(
                quantity=("quantity", "sum"),
                total_cost=("total_cost", "sum"),
                last_price=("last_price", "last"),
            )
            holdings_df["avg_price"] = (
                holdings_df["total_cost"] / holdings_df["quantity"]
            )
        else:
            holdings_df = new_holdings

    # Process executed sells efficiently
    executed_sells = orders_df.loc[
        (orders_df["action"] == "Sell") & (orders_df["executed"] == 1)
    ]
    if not executed_sells.empty:
        sell_agg = executed_sells.groupby(
            ["ticker", "strategy_name"], as_index=False
        ).agg(quantity=("quantity", "sum"))

        # Efficient merging using NumPy operations
        if holdings_df is not None and not holdings_df.empty:
            holdings_df = holdings_df.merge(
                sell_agg, on=["ticker", "strategy_name"], how="left"
            ).fillna({"quantity_y": 0})
            holdings_df["quantity"] = np.maximum(
                0, holdings_df["quantity"] - holdings_df["quantity_y"]
            )
            holdings_df.drop(columns=["quantity_y"], inplace=True)

            # Remove positions with zero quantity
            holdings_df = holdings_df.loc[holdings_df["quantity"] > 0]

            # Recalculate total cost & avg price efficiently
            holdings_df["total_cost"] = (
                holdings_df["avg_price"] * holdings_df["quantity"]
            )

    return holdings_df


# ai
def update_holdings_df_from_orders(holdings_df, orders_df):
    """
    Vectorized update of holdings DataFrame based on executed orders in orders_df.
    Args:
        holdings_df (pd.DataFrame): Current holdings, with columns ['ticker', 'strategy_name', 'quantity', 'avg_price', ...].
        orders_df (pd.DataFrame): Orders with columns ['ticker', 'strategy_name', 'action', 'quantity', 'current_price', 'executed', ...].
    Returns:
        pd.DataFrame: Updated holdings DataFrame.
    """
    # Only process executed orders
    executed = orders_df[orders_df["executed"] == 1].copy()
    if executed.empty:
        return holdings_df

    # Ensure required columns
    for col in ["ticker", "strategy_name", "quantity", "current_price"]:
        if col not in executed.columns:
            raise ValueError(f"orders_df missing required column: {col}")

    # Split buys and sells
    buys = executed[executed["action"] == "Buy"].copy()
    sells = executed[executed["action"] == "Sell"].copy()

    # --- Process Sells ---
    if not holdings_df.empty and not sells.empty:
        # Merge to match sells to holdings
        merged = pd.merge(
            holdings_df,
            sells[["ticker", "strategy_name", "quantity"]],
            on=["ticker", "strategy_name"],
            how="left",
            suffixes=("", "_sell"),
        )
        merged["quantity_sell"] = merged["quantity_sell"].fillna(0)
        merged["quantity"] = merged["quantity"] - merged["quantity_sell"]
        # Remove holdings with zero or negative quantity
        holdings_df = merged[merged["quantity"] > 0][holdings_df.columns]

    # --- Process Buys ---
    if not buys.empty:
        # Group by ticker/strategy for total quantity and weighted avg price
        buys["total_cost"] = buys["quantity"] * buys["current_price"]
        grouped = (
            buys.groupby(["ticker", "strategy_name"])
            .agg(
                quantity=("quantity", "sum"),
                total_cost=("total_cost", "sum"),
                stop_loss=("stop_loss", "last"),
                take_profit=("take_profit", "last"),
            )
            .reset_index()
        )
        grouped["avg_price"] = grouped["total_cost"] / grouped["quantity"]

        # Merge buys with existing holdings
        if holdings_df.empty:
            new_holdings = grouped[
                [
                    "ticker",
                    "strategy_name",
                    "quantity",
                    "avg_price",
                    "stop_loss",
                    "take_profit",
                ]
            ]
        else:
            merged = pd.merge(
                holdings_df,
                grouped,
                on=["ticker", "strategy_name"],
                how="outer",
                suffixes=("_old", "_buy"),
            )
            merged["quantity_old"] = merged["quantity_old"].fillna(0)
            merged["avg_price_old"] = merged["avg_price_old"].fillna(0)
            merged["quantity_buy"] = merged["quantity_buy"].fillna(0)
            merged["avg_price_buy"] = merged["avg_price_buy"].fillna(0)
            merged["stop_loss_buy"] = merged["stop_loss_buy"].combine_first(
                merged["stop_loss_old"]
            )
            merged["take_profit_buy"] = merged[
                "take_profit_buy"
            ].combine_first(merged["take_profit_old"])

            # Weighted average price for buys
            merged["quantity"] = (
                merged["quantity_old"] + merged["quantity_buy"]
            )
            merged["avg_price"] = (
                (
                    merged["quantity_old"] * merged["avg_price_old"]
                    + merged["quantity_buy"] * merged["avg_price_buy"]
                )
                / merged["quantity"]
            ).where(merged["quantity"] > 0, 0)

            new_holdings = merged.loc[
                merged["quantity"] > 0,
                [
                    "ticker",
                    "strategy_name",
                    "quantity",
                    "avg_price",
                    "stop_loss_buy",
                    "take_profit_buy",
                ],
            ].rename(
                columns={
                    "stop_loss_buy": "stop_loss",
                    "take_profit_buy": "take_profit",
                }
            )

        holdings_df = new_holdings.reset_index(drop=True)

    return holdings_df


if __name__ == "__main__":
    # Get the current filename without extension
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f"log/{module_name}.log"
    # Clear the log file at the start of each run
    with open(log_filename, "w"):
        pass
    LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename
    LOG_CONFIG["handlers"]["file_dynamic"]["level"] = "INFO"

    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)

    # setup database connections
    strategy_decisions_final_db_name = os.path.join(
        "PriceData", "strategy_decisions_final.db"
    )
    trading_account_db_name = os.path.join("PriceData", "trading_account.db")
    PRICE_DB_PATH = os.path.join("PriceData", "price_data.db")
    trades_list_db_name = os.path.join(
        "PriceData", "trades_list_vectorised.db"
    )

    # setup dates
    start_date = datetime.strptime(test_period_start, "%Y-%m-%d")
    # test_period_end = "2025-01-06"
    end_date = datetime.strptime(test_period_end, "%Y-%m-%d")

    # Create a US business day calendar
    us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # Generate the date range using the custom business day calendar
    test_date_range = pd.bdate_range(
        start=start_date, end=end_date, freq=us_business_day
    )
    # logger.info(f"Testing period: {start_date} to {end_date}")
    logger.info(f"{test_period_start = } {test_period_end = }")

    # Initialize testing variables
    # strategies = [strategies[2]]
    # strategies = strategies_top10_acc
    strategies = [strategies_top10_acc[1]]  # ULTOSC
    # strategies = [strategies_top10_acc[1], strategies_top10_acc[2]]  # ULTOSC
    tickers_list = train_tickers
    account = initialize_test_account(train_start_cash)
    # prediction_threshold = 0.75 #use global config
    use_rf_model_predictions = True
    train_rf_classifier = True
    # experiment_name = f"{use_rf_model_predictions = }_{len(train_tickers)}_{test_period_start}_{test_period_end}_{train_stop_loss}_{train_take_profit}_thres{prediction_threshold}"
    experiment_name = f"baseline_fees_{test_period_start}_{test_period_end}"
    account_values = pd.Series(
        index=pd.date_range(start=start_date, end=end_date)
    )
    rf_dict = {}

    # TODO? delete existing experiment tables from trading_account db?

    # prepare regime features data. Maybe just update ticker_price_history dict in memory?
    try:
        with sqlite3.connect(PRICE_DB_PATH) as conn:
            _ = prepare_sp500_one_day_return(conn)
    except Exception as e:
        logger.error(
            f"Error preparing regime featrure sp500_one_day_return {PRICE_DB_PATH}: {e}"
        )

    # get price history for tickers
    ticker_price_history = {}
    try:
        with sqlite3.connect(PRICE_DB_PATH) as conn:
            for ticker in train_tickers + regime_tickers:
                query = (
                    f"SELECT * FROM '{ticker}' WHERE Date >= ? AND Date <= ?"
                )
                ticker_price_history[ticker] = pd.read_sql(
                    query,
                    conn,
                    params=(start_date, end_date),
                    index_col=["Date"],
                )
    except Exception as e:
        logger.error(f"Error getting price data from {PRICE_DB_PATH}: {e}")

    # get strategy decisions from strategy_decisions_final.db
    precomputed_decisions = {}
    try:
        with sqlite3.connect(strategy_decisions_final_db_name) as conn:
            for idx, strategy in enumerate(strategies):
                strategy_name = strategy.__name__
                # query = f"SELECT * FROM '{strategy_name}' WHERE Date >= ? AND Date <= ?"
                query = f"SELECT * FROM '{strategy_name}' "
                precomputed_decisions[strategy_name] = pd.read_sql(
                    query,
                    conn,
                    # params=(start_date, end_date),
                    index_col=["Date"],
                )
            # logger.info(f"{precomputed_decisions = }")
    except Exception as e:
        logger.error(
            f"Error getting strategy decisions from {strategy_decisions_final_db_name}: {e}"
        )

    # logger.info(f"{ticker_price_history=}")
    account = main_test_loop(
        account,
        test_date_range,
        tickers_list,
        ticker_price_history,
        precomputed_decisions,
        use_rf_model_predictions,
        train_rf_classifier,
        account_values,
        trading_account_db_name,
        rf_dict,
        experiment_name,
        train_trade_liquidity_limit,
        logger,
    )

    # experiment global settings summary
    logger.info(f"Experiment settings:")
    logger.info(f"{experiment_name = }")
    logger.info(f"{test_period_start = } {test_period_end = }")
    strategy_names_list = []
    for strategy in strategies:
        strategy_names_list.append(strategy.__name__)
    logger.info(f"{len(strategy_names_list)=} {strategy_names_list=}")
    logger.info(f"{train_start_cash=} {train_trade_liquidity_limit=}")
    logger.info(f"{train_stop_loss=} {train_take_profit=}")
    logger.info(f"{train_trade_asset_limit=} {minimum_cash_allocation=}")
    logger.info(f"{len(train_tickers)=}{train_tickers=}")
    logger.info(f"{use_rf_model_predictions=} {prediction_threshold=}")
    logger.info(f"{rf_dict=}")
    logger.info(f"{regime_tickers=}")
    logger.info(f"{benchmark_asset=}")
    logger.info(f"{trading_account_db_name=}")

    # log trades to db
