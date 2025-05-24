import heapq
import logging
import logging.config
import os
from pyexpat import features
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
from PriceData.walk_forward_backtest import (
    get_oscillator_features,
    prepare_feature_return_data,
)
from config import environment
from control import (
    benchmark_asset,
    minimum_cash_allocation,
    prediction_threshold,
    score_threshold,
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
    miniumim_training_data_length,
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
    account = {
        "holdings": {
            "AAPL": {
                "quantity": 10,
                "price": 100,
                "current_price": 150.0,
                "current_value": 1500.0,
            },
            "MSFT": {
                "quantity": 5,
                "price": 200,
                "current_price": 250.0,
                "current_value": 1250.0,
            },
        },
        "cash": 10000,
        "total_portfolio_value": 12750.0,
    }
    """

    return {
        "holdings": {},
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
    ticker,
    quantity,
    current_price,
    stop_loss_pct,
    take_profit_pct,
):
    """
    Executes buy orders from the buy and suggestion heaps.
    Creates stop-loss and take-profit prices.

    Parameters:
    - account (dict): The account state.
    - action (str): The action to perform ("Buy").
    - ticker (str): The stock ticker.
    - quantity (float): Number of shares to buy.
    - current_price (float): Current price per share.
    - stop_loss_pct (float): Stop loss percentage (e.g., 0.05 for 5%).
    - take_profit_pct (float): Take profit percentage (e.g., 0.10 for 10%).

    Returns:
    - dict: The updated account state after executing the buy orders.
    """
    if action == "Buy":
        # update account qty, sl, tp, and cash
        account["cash"] -= round(quantity * current_price, 2)
        account["cash"] = round(account["cash"], 2)

        if ticker not in account["holdings"]:
            account["holdings"][ticker] = {
                "quantity": 0,
                "price": 0,
                "stop_loss": 0,
                "take_profit": 0,
            }

        account["holdings"][ticker]["quantity"] += round(quantity, 2)
        account["holdings"][ticker]["price"] = current_price
        account["holdings"][ticker]["stop_loss"] = round(
            current_price * (1 - stop_loss_pct), 2
        )
        account["holdings"][ticker]["take_profit"] = round(
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
    # strategy_name,
    account,
    current_price,
    current_date,
    note,
    logger,
):
    if action.lower() == "sell" and ticker in account["holdings"]:
        quantity = account["holdings"][ticker].get("quantity", 0)
        if quantity <= 0:
            logger.warning(f"No shares to sell for {ticker}.")
            return account

        account["cash"] += quantity * current_price
        account["cash"] = round(account["cash"], 2)

        del account["holdings"][ticker]

        try:
            account = update_account_fees_per_order(
                account, quantity, current_price
            )
        except Exception as e:
            logger.error(f"error update_account_fees_per_order {e}")

        # logger.info(
        #     f"{ticker} - Sold {quantity} shares at ${current_price} for {strategy_name} date: {current_date.strftime('%Y-%m-%d')}"
        # )
        logger.info(
            f"{ticker} - Sold {quantity} shares at ${current_price} date: {current_date.strftime('%Y-%m-%d')}"
        )
    else:
        logger.warning(
            f"Sell action ignored: action={action}, ticker={ticker} not in holdings."
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


# new allocation method
def ticker_cash_allocation(
    buy_orders_df,
    account,
    score_threshold,
    train_trade_asset_limit,
    prediction_threshold,
    trade_liquidity_limit_cash,
    minimum_cash_allocation,
    logger,
):
    if buy_orders_df.empty:
        logger.info("buy_orders_df is empty.")
        return pd.DataFrame()

    buy_df = buy_orders_df.copy()
    # Filter for probability > 0.6
    buy_df = buy_df[buy_df["probability"] > prediction_threshold]

    if buy_df.empty:
        logger.info(
            f"No buy orders with probability > {prediction_threshold=}."
        )
        return pd.DataFrame()

    buy_df["score"] = buy_df["prediction"] * buy_df["probability"]

    # Aggregate by ticker: sum scores for each ticker
    agg_df = (
        buy_df.groupby("ticker", as_index=False)
        .agg(
            score=("score", "sum"),
            current_price=("current_price", "last"),
            date=("date", "last"),
            action=("action", "first"),
        )
        .sort_values(by="score", ascending=False)
    )

    # Filter tickers by score_threshold
    agg_df["score"] = round(agg_df["score"], 2)
    agg_df = agg_df[agg_df["score"] > score_threshold].copy()
    if agg_df.empty:
        logger.info("No tickers exceed the score_threshold.")
        return agg_df

    # Allocate 10% of portfolio value to each qualifying ticker, but do not exceed available cash
    num_tickers = len(agg_df)
    if num_tickers == 0:
        logger.info("No tickers to allocate cash to.")
        return agg_df
    portfolio_value = account.get("total_portfolio_value", 0)
    available_cash = account.get("cash", 0) - trade_liquidity_limit_cash

    if available_cash <= 0 or portfolio_value <= 0:
        logger.info(
            f"Not enough available cash: {available_cash} or portfolio value is zero: {portfolio_value}"
        )
        return pd.DataFrame()

    max_allocation_per_ticker = min(
        train_trade_asset_limit * portfolio_value, available_cash / num_tickers
    )

    if max_allocation_per_ticker <= minimum_cash_allocation:
        logger.info(
            "Max allocation per ticker is less than minimum_cash_allocation, no buys."
        )
        return pd.DataFrame()

    agg_df["allocated_cash"] = round(max_allocation_per_ticker, 2)
    agg_df["allocation_pct"] = (
        agg_df["allocated_cash"] / portfolio_value if portfolio_value else 0
    )

    agg_df["quantity"] = agg_df["allocated_cash"] / agg_df["current_price"]
    agg_df["quantity"] = round(agg_df["quantity"], 2)

    return agg_df


def update_account_portfolio_values(
    account, ticker_price_history, current_date
):
    """
    Updates the account dictionary with the latest portfolio values based on current prices.

    Updates each holding with the latest current_price and recalculates its current_value.
    Aggregates the value of holdings by ticker in holdings_value_by_ticker.
    Computes the total portfolio value as the sum of cash and all holdings.

    Args:
        account (dict): The account data containing 'cash', 'holdings', and other info.
        ticker_price_history (dict): Mapping of ticker symbols to their historical price DataFrames.
        current_date (datetime): The date for which to fetch the latest closing prices.

    Returns:
        dict: The updated account dictionary with:
            - 'holdings_value_by_ticker': Value of holdings grouped by ticker.
            - 'total_portfolio_value': Total value of the portfolio (cash + holdings).
            - Updates to each holding with 'current_price' and 'current_value'.

    Notes:
        - Assumes each holding contains a 'quantity' field.
        - Expects price DataFrames to have a 'Close' column and be indexed by date strings.
    """

    # Calculate and update total portfolio value
    total_value = account["cash"]
    account["holdings_value_by_ticker"] = {}
    for ticker, holding in account["holdings"].items():
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
        # value of each holding
        current_value = round(holding["quantity"] * current_price, 2)
        holding["current_value"] = current_value

        # value of holdings by ticker
        account["holdings_value_by_ticker"][ticker] = current_value

        # total value of portfolio
        total_value += current_value
    account["total_portfolio_value"] = round(total_value, 2)

    return account


def insert_trade_into_trading_account_db(
    trades_df, trading_account_db_name, experiment_name, table_suffix
):
    """
    Inserts trades into the trading account database.

    Parameters:
    - trades_df (pd.DataFrame): DataFrame containing trade data.
    - trading_account_db_name (str): path to the trading account database.
    - experiment_name (str): Name of the experiment for which the trades are being inserted.
    - table_suffix (str): first part of the table name

    Returns:
    - Boolean: success of insert True/False.
    """
    # Ensure the DataFrame is not empty
    if trades_df.empty:
        logger.warning("No data to insert into the database.")
        return

    # order_id check if column name exists
    if "order_id" not in trades_df.columns:
        # Add a new column with the order_id value
        if "strategy_name" not in trades_df.columns:
            trades_df["order_id"] = (
                trades_df["ticker"] + "_" + trades_df["date"].astype(str)
            )

        # prediction df still has strategy name
        if "strategy_name" in trades_df.columns:
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
                f"{table_suffix}_{experiment_name}",
                conn,
                if_exists="append",
                index=True,
                dtype={"order_id": "TEXT PRIMARY KEY"},
            )
        return True
    except Exception as e:
        logger.error(f"Error saving trades to database: {e}")
        logger.error(f"Error saving trades_df:\n{trades_df}")
        return False


def summarize_account_tickers_by_value(account, top_n=10):
    """
    Returns a summary of the top N tickers in the account by holding value.

    Args:
        account (dict): The account data containing 'holdings' keyed by ticker.
        top_n (int): Number of top tickers to return.

    Returns:
        list of tuples: Each tuple is (ticker, current_value), sorted by value descending.
    """
    ticker_values = []
    for ticker, holding in account.get("holdings", {}).items():
        value = holding.get("current_value", 0)
        ticker_values.append((ticker, value))
    # Sort by value descending
    ticker_values.sort(key=lambda x: x[1], reverse=True)
    return ticker_values[:top_n]


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

        # agg sell_orders_df by ticker. possibly multiple sell orders for same ticker.
        sell_orders_df = sell_orders_df.groupby("ticker", as_index=False).agg(
            current_price=("current_price", "last"),
            date=("date", "last"),
            action=("action", "first"),
            note=("note", "first"),
            # strategy_name=("strategy_name", "first"),
        )

        for index, row in sell_orders_df.iterrows():
            try:
                account = execute_sell_orders(
                    row["action"],
                    row["ticker"],
                    # row["strategy_name"],
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
        buy_df = ticker_cash_allocation(
            buy_orders_df,
            account,
            score_threshold,
            train_trade_asset_limit,
            prediction_threshold,
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

    cols_to_keep = [
        "ticker",
        "action",
        "current_price",
        "date",
        "note",
        "allocated_cash",
        "quantity",
        "score",
        "executed",
    ]
    orders_df = orders_df[cols_to_keep]

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
    features_df=None,
):
    """
    Design desicions:
        - If have position (qty>0): skip subsequent buy orders?
        (otherwise fills up any remaining allocation with small orders).

        - Minimum order size?

        - combine execute buy and sell into one execute orders function?
        pros: simpler logic.
    """
    if train_rf_classifier:
        if features_df is None:
            logger.error(
                "features_df must be provided when train_rf_classifier is True."
            )
            return

    for date in test_date_range:
        logger.info(
            f"\n\n\n------------- NEW INTERVAL {date.strftime("%Y-%m-%d")}"
        )
        orders_list = []  # combine buy and sell orders into one list
        predictions_list = []
        training_data_checked = {}
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

            historic_trades_df = pd.DataFrame()
            # prepared_training_data = False
            prediction = None
            accuracy = None
            probability = None
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
                    account.get("holdings", {}).get(ticker, {})
                    # .get(strategy_name, {})
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
                            "note": "Sell signal: " + strategy_name,
                            "date": date.strftime("%Y-%m-%d"),
                        }
                    )
                    continue

                # logger.info(f"{account = }")

                elif action == "Buy" and quantity == 0:

                    # only load rf_model if buy signal
                    if use_rf_model_predictions:
                        """
                        for each date in date_list
                        loop through strategy_list:
                        loop through ticker_list:
                        if ticker action == "Buy"
                        Once per date, for each strategy, check historical training data length, if not checked already.
                        if historical training data length has changed, use it to retrain the rf model.
                        Make predictions and store in rf_dict[strategy_name].
                        if next ticker requires prediction, ie action == "Buy", then use the prediction already stored in rf_dict[strategy_name].

                        """
                        # Check historical training data length, if not checked
                        if strategy_name not in training_data_checked:
                            logger.info("Get training data for strategy...")
                            historic_trades_df = get_training_data(
                                strategy_name,
                                features_df,
                                trades_list_db_name,
                                date,
                                logger,
                                start_date=None,
                            )

                            if (
                                historic_trades_df is not None
                                and len(historic_trades_df)
                                < miniumim_training_data_length
                            ):
                                logger.warning(
                                    f"historic_trades_df below threshold (not enough training data) {len(historic_trades_df)=}"
                                )
                                break
                            if historic_trades_df is not None:
                                if strategy_name not in rf_dict:
                                    logger.info(
                                        f"Very first get training data for strategy, initialising rf_dict"
                                    )
                                    rf_dict[strategy_name] = {}
                                    rf_dict[strategy_name][
                                        "prev_len(historic_trades_df)"
                                    ] = 0
                                    rf_dict[strategy_name][
                                        "len(historic_trades_df)"
                                    ] = 0

                                rf_dict[strategy_name][
                                    "len(historic_trades_df)"
                                ] = len(historic_trades_df)
                                # logger.info(f"{rf_dict=}")

                                training_data_checked[strategy_name] = len(
                                    historic_trades_df
                                )
                                logger.info(
                                    f"training data len: {rf_dict[strategy_name][
                                        "prev_len(historic_trades_df)"
                                    ]} -> {rf_dict[strategy_name][
                                        "len(historic_trades_df)"
                                    ]}"
                                )

                        # If historical training data length has changed, retrain RF model
                        if (
                            training_data_checked[strategy_name]
                            != rf_dict[strategy_name][
                                "prev_len(historic_trades_df)"
                            ]
                            and historic_trades_df is not None
                        ):
                            logger.info(
                                f"training data has changed, retrain model. {rf_dict[strategy_name][
                                        "prev_len(historic_trades_df)"
                                    ]} -> {training_data_checked[strategy_name]}"
                            )
                            try:
                                (
                                    rf_classifier,
                                    accuracy,
                                    precision,
                                    recall,
                                ) = train_random_forest_classifier_features(
                                    historic_trades_df,
                                    required_features,
                                )

                                # rf_dict[strategy_name] = {}
                                rf_dict[strategy_name][
                                    "rf_classifier"
                                ] = rf_classifier
                                rf_dict[strategy_name]["accuracy"] = accuracy
                                rf_dict[strategy_name]["precision"] = precision
                                rf_dict[strategy_name]["recall"] = recall
                                rf_dict[strategy_name][
                                    "prev_len(historic_trades_df)"
                                ] = len(historic_trades_df)

                                logger.info(
                                    f"successfully trained new rf model"
                                )
                            except Exception as error:
                                logger.error(
                                    f"error training rf model {error}"
                                )

                        # make prediction using features_df
                        if prediction is None and features_df is not None:
                            logger.info("getting predictions...")
                            # Get prediction (0 or 1) and probability of class 1 (positive return)
                            raw_prediction, raw_probability, probabilities = (
                                predict_random_forest_classifier(
                                    rf_dict[strategy_name]["rf_classifier"],
                                    features_df.loc[
                                        [date.strftime("%Y-%m-%d")]
                                    ],
                                )
                            )
                            prediction = raw_prediction[0]

                            accuracy = round(
                                rf_dict[strategy_name]["accuracy"], 2
                            )  # Keep accuracy for logging/potential future use
                            probability = np.round(raw_probability[0], 4)

                        if prediction == 0:
                            action = "Hold"  # Override original 'Buy' if RF doesn't confirm

                        # only log +ve predictions
                        if prediction == 1:
                            predictions_list.append(
                                {
                                    "strategy_name": strategy_name,
                                    "ticker": ticker,
                                    "action": action,
                                    "prediction": prediction,
                                    "accuracy": accuracy,
                                    "probability": probability,
                                    "current_price": current_price,
                                    "date": date.strftime("%Y-%m-%d"),
                                }
                            )

                        logger.info(
                            f"Prediction {date.strftime('%Y-%m-%d')} {strategy_name} {ticker}: {prediction} Prob: {probability:.4f}, Acc: {accuracy}, Action: {action}"
                        )

                    # create order if action still buy after rf prediction.
                    if action == "Buy":
                        if not use_rf_model_predictions:
                            # used if use_rf_model_predictions == False. ie baseline test.
                            prediction = 1
                            accuracy = 1
                            probability = 1

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

            # daily orders summary

            # TODO: save predictions to db (including prediction = 1)
            if use_rf_model_predictions:
                predictions_df = pd.DataFrame(predictions_list)
                insert_trade_into_trading_account_db(
                    predictions_df,
                    trading_account_db_name,
                    experiment_name,
                    "predict",
                )

            # convert orders_list into a df
            orders_df = pd.DataFrame(orders_list)

            # Required columns for orders
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
                # features
                if features_df is not None and not features_df.empty:
                    logger.info(
                        f"current features: {features_df.loc[[date.strftime("%Y-%m-%d")]]}"
                    )
                # summarize orders: use orders_df to show number of buy and sell orders by strategy
                # orders_summary = orders_df.groupby(
                #     ["strategy_name", "action"]
                # ).size()
                # logger.info(f"Orders summary:\n{orders_summary}")

                orders_summary = orders_df.groupby(
                    ["strategy_name", "action"]
                ).agg(
                    {
                        "probability": "mean",
                        "accuracy": "mean",
                    }
                )
                logger.info(f"Orders summary:\n{orders_summary}")

                # execute orders and update account
                logger.info(f"pre processing orders_df\n{orders_df}")
                account, orders_df = process_orders(
                    orders_df,
                    account,
                    date,
                    train_trade_liquidity_limit,
                    train_stop_loss,
                    train_take_profit,
                    logger,
                )
                logger.info(f"post processing orders_df\n{orders_df}")

                # Calculate and update account total portfolio value
                account = update_account_portfolio_values(
                    account, ticker_price_history, date
                )
                logger.info(f"orders_df:\n\n{orders_df}")

                insert_trade_into_trading_account_db(
                    orders_df,
                    trading_account_db_name,
                    experiment_name,
                    "trades",
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
            # logger.info(f"holdings_value_by_strategy:")
            # logger.info(account["holdings_value_by_strategy"])
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


def get_training_data(
    strategy_name,
    features_df,
    trades_list_db_name,
    date,
    logger,
    start_date=None,
):
    # logger.info("preparing training data...")
    # Load trades data (training data)
    # Get historical trades from up to one day before
    previous_day = date - timedelta(days=1)
    historic_trades_df = get_trades_training_data_from_db(
        strategy_name,
        trades_list_db_name,
        logger,
        None,
        previous_day.strftime("%Y-%m-%d"),
    )

    if historic_trades_df.empty:
        return historic_trades_df

    # join features_df to trades_df on buy_date

    # filter features_df by previous date
    historic_features_df = features_df[
        features_df.index <= previous_day.strftime("%Y-%m-%d")
    ]

    # HACK drop ^VIX from trades_df to avoid duplicate column names
    historic_trades_df.drop(
        columns=["^VIX"],
        inplace=True,
        errors="ignore",
    )

    historic_trades_df = historic_trades_df.join(
        historic_features_df[required_features],
        on="buy_date",
        how="left",
    )

    # check for missing features
    missing_features = [
        f for f in required_features if f not in historic_trades_df.columns
    ]
    if missing_features:
        logger.error(
            f"Missing required features after join for {strategy_name}: {missing_features}. Skipping."
        )
        return
    # logger.info(f"{len(historic_trades_df)=}")
    # logger.info(f"{historic_trades_df=}")
    assert (
        historic_trades_df["sell_date"].max() < date
    ), "historic_trades_df date error"
    return historic_trades_df


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


def check_strategies_have_historical_trades(
    strategies, trades_list_db_name, logger
):
    # get list of tables in trades_list_db_name
    with sqlite3.connect(trades_list_db_name) as conn:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, conn)
        logger.info(
            f"Tables in {trades_list_db_name}: {len(tables)=} {tables['name'].tolist()}"
        )

    # Build a new list instead of removing while iterating
    valid_strategies = []
    strategies_not_in_db = []
    for strategy in strategies:
        strategy_name = strategy.__name__
        if strategy_name in tables["name"].tolist():
            valid_strategies.append(strategy)
        else:
            # logger.warning(
            #     f"Strategy {strategy_name} not found in {trades_list_db_name}. Removing from strategy array."
            # )
            strategies_not_in_db.append(strategy_name)

    logger.info(
        f"{len(strategies)=}, {len(valid_strategies)=}, {len(strategies_not_in_db)=}"
    )
    return valid_strategies


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
    trading_account_db_name = os.path.join(
        "PriceData", f"trading_account_{environment}.db"
    )
    if os.path.exists(trading_account_db_name):
        os.remove(trading_account_db_name)
        logger.warning(f"deleted {trading_account_db_name}")

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
    strategies = strategies
    # strategies = strategies_top10_acc
    # strategies = [strategies_top10_acc[1]]  # ULTOSC
    # strategies = [strategies_top10_acc[1], strategies_top10_acc[2]]  # ULTOSC
    tickers_list = train_tickers
    account = initialize_test_account(train_start_cash)
    # prediction_threshold = 0.75 #use global config
    use_rf_model_predictions = True
    train_rf_classifier = True
    # experiment_name = f"{use_rf_model_predictions = }_{len(train_tickers)}_{test_period_start}_{test_period_end}_{train_stop_loss}_{train_take_profit}_thres{prediction_threshold}"
    experiment_name = f"rf_fees_07prob_{test_period_start}_{test_period_end}"
    account_values = pd.Series(
        index=pd.date_range(start=start_date, end=end_date)
    )
    rf_dict = {}

    # Check historical trades list. Remove strategies with no trades.
    strategies = check_strategies_have_historical_trades(
        strategies, trades_list_db_name, logger
    )

    # Regime Data
    # prepare regime features data. Maybe just update ticker_price_history dict in memory?
    periods_list = [1, 5, 10, 20, 30]
    features_ticker_list = ["^GSPC"]
    oscillator_features_ticker_list = ["^VIX"]
    required_features = []
    for ticker in features_ticker_list:
        for period in periods_list:
            required_features.append(f"{ticker}_return({period})")
    for ticker in oscillator_features_ticker_list:
        required_features.append(ticker)

    features_df = pd.DataFrame()
    features_df = prepare_feature_return_data(
        features_ticker_list, periods_list, PRICE_DB_PATH, logger
    )

    for ticker in oscillator_features_ticker_list:
        oscillator_features_df = get_oscillator_features(ticker, PRICE_DB_PATH)
        # Join to trades_df on buy_date
        features_df = features_df.join(
            oscillator_features_df, on="Date", how="left"
        )
    logger.info(f"{features_df=}")

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
                    params=(test_period_start, test_period_end),
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
        features_df,
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
    logger.info(f"{len(train_tickers)=} {train_tickers=}")
    logger.info(f"{use_rf_model_predictions=} {prediction_threshold=}")
    logger.info(f"{rf_dict=}")
    logger.info(f"{regime_tickers=}")
    logger.info(f"{benchmark_asset=}")
    logger.info(f"{trading_account_db_name=}")

    # log trades to db
