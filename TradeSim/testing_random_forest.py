import heapq
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
import sqlite3
import logging

import certifi
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pymongo import MongoClient

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper_files.client_helper import setup_logging
from random_forest import predict_random_forest_classifier, train_and_store_classifiers
from PriceData.store_rf_models import get_tables_list, check_model_exists, load_rf_model


# from variables import config_dict
from TradeSim.variables import config_dict

import wandb
from config import FINANCIAL_PREP_API_KEY, mongo_url, PRICE_DB_PATH
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

from helper_files.client_helper import (
    get_ndaq_tickers,
    strategies_top10_acc,
    strategies,
)
from helper_files.train_client_helper import (
    calculate_metrics,
    generate_tear_sheet,
    local_update_portfolio_values,
)
from TradeSim.utils import (
    compute_trade_quantities,
    compute_trade_quantities_only_buy_one,
    simulate_trading_day,
    update_time_delta,
)
from trading_client import weighted_majority_decision_and_median_quantity

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

ca = certifi.where()


# Check if the module is being run as part of a unit test
# if "unittest" not in sys.modules:
#     mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
mongo_client = MongoClient(mongo_url, tlsCAFile=ca)


def initialize_test_account():
    """
    Initializes the test trading account with starting parameters
    """
    return {
        "holdings": {},
        "cash": train_start_cash,
        "trades": [],
        "total_portfolio_value": train_start_cash,
    }


def check_stop_loss_take_profit(
    account, ticker, current_price, current_date, trading_account_db_name
):
    """
    Checks and executes stop loss and take profit orders for a given ticker.

    Parameters:
    - account (dict): The current account state, including holdings, cash, and trades.
    - ticker (str): The ticker symbol of the asset to check.
    - current_price (float): The current price of the asset.

    Returns:
    - dict: The updated account state after executing stop loss and take profit orders.
    """
    if ticker in account["holdings"]:
        strategies_to_remove = []
        for strategy, holding in account["holdings"][ticker].items():
            if holding["quantity"] > 0:
                if current_price < holding["stop_loss"]:
                    # account["trades"].append(
                    #     {
                    #         "symbol": ticker,
                    #         "quantity": holding["quantity"],
                    #         "price": current_price,
                    #         "action": "sell - stop_loss",
                    #         "strategy": strategy,
                    #         "date": current_date.strftime("%Y-%m-%d"),
                    #     }
                    # )

                    trade_id_value = (
                        f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}"
                    )

                    trade_df = pd.DataFrame(
                        {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "symbol": ticker,
                            "action": "sell - stop_loss",
                            "quantity": holding["quantity"],
                            "price": round(current_price, 2),
                            "total_value": round(
                                holding["quantity"] * current_price, 2
                            ),
                            "strategy": strategy_name,
                        },
                        index=[trade_id_value],
                    )
                    trade_df.index.name = "trade_id"

                    insert_trade_into_tranding_account_db(
                        trade_df, trading_account_db_name, experiment_name
                    )

                    account["cash"] += holding["quantity"] * current_price
                    strategies_to_remove.append(strategy)
                elif current_price > holding["take_profit"]:
                    # account["trades"].append(
                    #     {
                    #         "symbol": ticker,
                    #         "quantity": holding["quantity"],
                    #         "price": current_price,
                    #         "action": "sell - take_profit",
                    #         "strategy": strategy,
                    #         "date": current_date.strftime("%Y-%m-%d"),
                    #     }
                    # )

                    trade_id_value = (
                        f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}"
                    )

                    trade_df = pd.DataFrame(
                        {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "symbol": ticker,
                            "action": "sell - take_profit",
                            "quantity": holding["quantity"],
                            "price": round(current_price, 2),
                            "total_value": round(
                                holding["quantity"] * current_price, 2
                            ),
                            "strategy": strategy_name,
                        },
                        index=[trade_id_value],
                    )
                    trade_df.index.name = "trade_id"

                    insert_trade_into_tranding_account_db(
                        trade_df,
                        trading_account_db_name,
                        experiment_name,
                    )

                    account["cash"] += holding["quantity"] * current_price
                    strategies_to_remove.append(strategy)

        for strategy in strategies_to_remove:
            del account["holdings"][ticker][strategy]

        if not account["holdings"][ticker]:
            del account["holdings"][ticker]

    return account


def execute_buy_orders(
    buy_heap,
    suggestion_heap,
    account,
    current_date,
    train_trade_liquidity_limit,
    train_stop_loss,
    train_take_profit,
):
    """
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
    while (buy_heap or suggestion_heap) and float(
        account["cash"]
    ) > train_trade_liquidity_limit:
        if buy_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = buy_heap
        elif suggestion_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = suggestion_heap
        else:
            break

        _, quantity, ticker, current_price, strategy_name = heapq.heappop(heap)
        logger.info(f"Executing BUY order for {ticker} of quantity {quantity}")

        # account["trades"].append(
        #     {
        #         "symbol": ticker,
        #         "quantity": quantity,
        #         "price": current_price,
        #         "action": "buy",
        #         "date": current_date.strftime("%Y-%m-%d"),
        #         "strategy": strategy_name,
        #     }
        # )

        # insert trade into trading account database
        # Calculate trade_id first
        trade_id_value = f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}"

        trade_df = pd.DataFrame(
            {
                # "trade_id": f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}",
                "date": current_date.strftime("%Y-%m-%d"),
                "symbol": ticker,
                "action": "buy",
                "quantity": quantity,
                "price": current_price,
                "total_value": round(quantity * current_price, 2),
                "strategy": strategy_name,
            },
            index=[trade_id_value],
        )
        trade_df.index.name = "trade_id"

        insert_trade_into_tranding_account_db(
            trade_df, trading_account_db_name, experiment_name
        )

        account["cash"] -= quantity * current_price
        if ticker not in account["holdings"]:
            account["holdings"][ticker] = {}

        if strategy_name not in account["holdings"][ticker]:
            account["holdings"][ticker][strategy_name] = {
                "quantity": 0,
                "price": 0,
                "stop_loss": 0,
                "take_profit": 0,
            }

        account["holdings"][ticker][strategy_name]["quantity"] += round(quantity, 2)
        account["holdings"][ticker][strategy_name]["price"] = current_price
        account["holdings"][ticker][strategy_name]["stop_loss"] = round(
            current_price * (1 - train_stop_loss), 2
        )
        account["holdings"][ticker][strategy_name]["take_profit"] = round(
            current_price * (1 + train_take_profit), 2
        )

    return account


def execute_sell_orders(
    action, ticker, strategy_name, account, current_price, current_date, logger
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
        action == "sell"
        and ticker in account["holdings"]
        and strategy_name in account["holdings"][ticker]
    ):
        # quantity = max(quantity, 1)
        quantity = account["holdings"][ticker][strategy_name]["quantity"]
        # account["trades"].append(
        #     {
        #         "symbol": ticker,
        #         "quantity": quantity,
        #         "price": round(current_price, 2),
        #         "action": "sell",
        #         "strategy": strategy_name,
        #         "date": current_date.strftime("%Y-%m-%d"),
        #     }
        # )

        trade_id_value = f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}"

        trade_df = pd.DataFrame(
            {
                # "trade_id": f"{ticker}_{strategy_name}_{current_date.strftime('%Y-%m-%d')}",
                "date": current_date.strftime("%Y-%m-%d"),
                "symbol": ticker,
                "action": "sell",
                "quantity": quantity,
                "price": round(current_price, 2),
                "total_value": round(quantity * current_price, 2),
                "strategy": strategy_name,
            },
            index=[trade_id_value],
        )
        trade_df.index.name = "trade_id"

        insert_trade_into_tranding_account_db(
            trade_df, trading_account_db_name, experiment_name
        )

        account["cash"] += quantity * current_price
        del account["holdings"][ticker][strategy_name]
        logger.info(
            f"{ticker} - Sold {quantity} shares at ${current_price} for {strategy_name} date: {current_date.strftime('%Y-%m-%d')}"
        )
    return account


def update_strategy_ranks(strategies, points, trading_simulator):
    """
    Updates strategy rankings based on performance
    """
    rank = {}
    q = []
    for strategy in strategies:
        if points[strategy.__name__] > 0:
            score = (
                points[strategy.__name__] * 2
                + trading_simulator[strategy.__name__]["portfolio_value"]
            )
        else:
            score = trading_simulator[strategy.__name__]["portfolio_value"]

        heapq.heappush(
            q,
            (
                score,
                trading_simulator[strategy.__name__]["successful_trades"]
                - trading_simulator[strategy.__name__]["failed_trades"],
                trading_simulator[strategy.__name__]["amount_cash"],
                strategy.__name__,
            ),
        )

    coeff_rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank[strategy_name] = coeff_rank
        coeff_rank += 1

    return rank


def test_random_forest(
    ticker_price_history,
    ideal_period,
    mongo_client,
    precomputed_decisions,
    # strategies,
    logger,
):
    """
    Runs the testing phase of the trading simulator.
    """
    global train_tickers
    logger.info("Starting testing phase...")

    # train rf classifiers
    """
    already done?
    """
    try:
        # Load the trades data
        trades_data_df = pd.read_csv(
            os.path.join(results_dir, f"{config_dict['experiment_name']}_trades.csv")
        )

        # train random forest classifier based on training trades list
        trained_classifiers, strategies_with_enough_data = train_and_store_classifiers(
            trades_data_df, logger
        )

    except Exception as e:
        logger.error(f"Error training rf classifiers {e}")
        return

    # need to store classifiers as pickle .pkl file separately from metadata because it is not json serializable.
    # store_dict_as_json(trained_classifiers, f"{config_dict['experiment_name']}_trained_classifiers.json", results_dir, logger)

    # Get rank coefficients from database. needed?
    db = mongo_client.trading_simulator
    r_t_c = db.rank_to_coefficient
    rank_to_coefficient = {doc["rank"]: doc["coefficient"] for doc in r_t_c.find({})}
    logger.info("Rank coefficients retrieved from database.")

    # Initialize testing variables
    strategy_to_coefficient = {}
    account = initialize_test_account()
    points = {}  # Initialize points
    trading_simulator = {}  # Initialize trading_simulator
    time_delta = {}  # Initialize time_delta

    # Load saved results
    logger.info("Loading saved training results...")
    try:
        with open(
            os.path.join(results_dir, f"{config_dict['experiment_name']}.json"), "r"
        ) as json_file:
            results = json.load(json_file)
            trading_simulator = results["trading_simulator"]
            points = results["points"]
            time_delta = results["time_delta"]
        logger.info(
            f"Training results loaded successfully from {config_dict['experiment_name']}.json"
        )
    except Exception as e:
        logger.error(
            f"Error loading training results. Filename: {config_dict['experiment_name']}.json. Error: {e}"
        )

    # Initialize testing variables
    strategy_to_coefficient = {}
    account = initialize_test_account()

    # needed?
    rank = update_strategy_ranks(strategies, points, trading_simulator)

    start_date = datetime.strptime(test_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(test_period_end, "%Y-%m-%d")
    current_date = start_date
    account_values = pd.Series(index=pd.date_range(start=start_date, end=end_date))
    logger.info(f"Testing period: {start_date} to {end_date}")
    if not train_tickers:
        train_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
    while current_date <= end_date:
        logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")

        # Skip non-trading days
        if (
            current_date.weekday() >= 5
            or current_date.strftime("%Y-%m-%d")
            not in ticker_price_history[train_tickers[0]].index
        ):
            logger.info(
                f"Skipping {current_date.strftime('%Y-%m-%d')} (weekend or missing data)."
            )
            current_date += timedelta(days=1)
            continue

        # Update strategy coefficients. needed?
        for strategy in strategies:
            strategy_to_coefficient[strategy.__name__] = rank_to_coefficient[
                rank[strategy.__name__]
            ]
        # logger.info(f"Strategy coefficients updated: {strategy_to_coefficient}")

        # Process trading day
        buy_heap, suggestion_heap = [], []
        date_str = current_date.strftime("%Y-%m-%d")
        prediction_results = {}
        prediction_results_list = []
        for ticker in train_tickers:
            if date_str in ticker_price_history[ticker].index:
                daily_data = ticker_price_history[ticker].loc[date_str]
                current_price = daily_data["Close"]
                # logger.info(f"{ticker} - Current price: {current_price}")

                # Check stop loss and take profit
                account = check_stop_loss_take_profit(
                    account,
                    ticker,
                    current_price,
                    current_date,
                    trading_account_db_name,
                )

                # Get strategy decisions
                decisions_and_quantities = []
                portfolio_qty = account["holdings"].get(ticker, {}).get("quantity", 0)

                for strategy in strategies:
                    strategy_name = strategy.__name__

                    # Get precomputed strategy decision json
                    # action = precomputed_decisions[strategy_name][ticker].get(date_str)

                    # if action is None:
                    #     # Skip if no precomputed decision (should not happen if properly precomputed)
                    #     logger.warning(
                    #         f"No precomputed decision for {ticker}, {strategy_name}, {date_str}"
                    #     )
                    #     continue

                    # Get precomputed strategy decision from DataFrame
                    """
                    refactor to get from strategy decisions intermediate database
                    
                    """

                    action = precomputed_decisions[strategy_name][
                        (
                            precomputed_decisions[strategy_name]["Strategy"]
                            == strategy_name
                        )
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

                    account_cash = account["cash"]
                    total_portfolio_value = account["total_portfolio_value"]

                    # Get prediction from random forest classifier
                    if strategy_name in strategies_with_enough_data:
                        if action == "Buy":
                            daily_vix_df = ticker_price_history["^VIX"].loc[date_str][
                                "Close"
                            ]
                            data = {"current_vix": [daily_vix_df]}
                            sample_df = pd.DataFrame(data, index=[0])

                            """
                            Load rf model. Pre-compute from strategy_decisions_intermediate.db. create new db with test dates?
                            """

                            # Get prediction (0 or 1) and probability of class 1 (positive return)
                            prediction, probability = predict_random_forest_classifier(
                                rf_dict[strategy_name]["rf_classifier"],
                                sample_df,
                            )

                            if prediction != 1:
                                action = "hold"  # Override original 'Buy' if RF doesn't confirm

                            accuracy = round(
                                rf_dict[strategy_name]["accuracy"], 2
                            )  # Keep accuracy for logging/potential future use
                            probability = np.round(probability, 4)  # Round probability

                            logger.info(
                                f"Prediction {date.strftime('%Y-%m-%d')} {strategy_name} {ticker}: {prediction} (Prob: {probability:.4f}), Acc: {accuracy}, VIX: {daily_vix_df:.2f}, SPY: {One_day_spy_return:.2f}, Action: {action}"
                            )

                            # Only add to results if the original action was Buy and RF prediction is 1
                            if (
                                action == "Buy"
                            ):  # Check if action is still Buy (meaning RF predicted 1)
                                prediction_results_list.append(
                                    {
                                        "strategy_name": strategy_name,
                                        "ticker": ticker,
                                        "action": action,  # Should always be 'Buy' here
                                        "prediction": prediction,  # Should always be 1 here
                                        "accuracy": accuracy,  # Historical accuracy of the model
                                        "probability": probability,  # Probability of this specific prediction being 1
                                        "current_price": current_price,
                                    }
                                )

                        # execute sell orders
                        elif action == "sell":
                            account = execute_sell_orders(
                                action,
                                ticker,
                                strategy_name,
                                account,
                                current_price,
                                current_date,
                                logger,
                            )

        # Convert list of dictionaries to DataFrame
        prediction_results_df = pd.DataFrame(prediction_results_list)
        # Save prediction_results_df to CSV in results folder
        csv_file_path = os.path.join(results_dir, "prediction_results.csv")
        prediction_results_df.to_csv(csv_file_path, index=False)

        # Calculate holdings value by strategy. needed to ensure cash allocation constraints are met.
        holdings_value_by_strategy = get_holdings_value_by_strategy(
            account, ticker_price_history, current_date
        )
        print(f"{holdings_value_by_strategy = }")

        buy_df = strategy_and_ticker_cash_allocation(
            prediction_results_df,
            account,
            holdings_value_by_strategy,
            prediction_threshold,
            train_trade_asset_limit,
            train_trade_strategy_limit,
        )

        buy_heap = create_buy_heap(buy_df)

        # Execute buy orders, create sl and tp prices
        account = execute_buy_orders(
            buy_heap,
            suggestion_heap,
            account,
            current_date,
            train_trade_liquidity_limit,
            train_stop_loss,
            train_take_profit,
        )

        # Simulate ranking updates. Updates list of trades.
        trading_simulator, points = simulate_trading_day(
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
        )

        # logger.info(f"{points = }")
        # logger.info(f"{trading_simulator = }")
        # Update trading_simulator portfolio values
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, strategies, trading_simulator, ticker_price_history, logger
        )

        # Update time delta needed?
        # time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # Calculate and update account total portfolio value
        account = update_account_portfolio_values(
            account, ticker_price_history, current_date
        )

        # Update account values for metrics
        total_value = account["total_portfolio_value"]
        account_values[current_date] = total_value

        # Update rankings #needed?
        rank = update_strategy_ranks(strategies, points, trading_simulator)

        # Log daily results
        logger.info("-------------------------------------------------")
        logger.info(f"Account Cash: ${account['cash']: ,.2f}")
        # logger.info(f"Trades: {account['trades']}")
        logger.info(f"Holdings: {account['holdings']}")
        logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
        logger.info(f"Active Count: {active_count}")
        logger.info("-------------------------------------------------")

        current_date += timedelta(days=1)
        time.sleep(5)

    # Calculate final metrics and generate tear sheet
    metrics = calculate_metrics(account_values)
    wandb.log(metrics)

    logger.info("Final metrics calculated.")
    logger.info(metrics)

    try:
        generate_tear_sheet(account_values, filename=f"{benchmark_asset}_vs_strategy")
        logger.info("Tear sheet generated.")
    except Exception as e:
        logger.error(f"Error generating tear sheet: {e}")

    # Print final results
    logger.info("Testing Completed.")
    logger.info("-------------------------------------------------")
    logger.info(f"Account Cash: ${account['cash']: ,.2f}")
    logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
    logger.info("-------------------------------------------------")


def test_random_forest_v2(
    ticker_price_history,
    ideal_period,
    mongo_client,
    precomputed_decisions,
    # strategies,
    logger,
):
    """
    Runs the testing phase of the trading simulator.
        action = precomputed_decisions
    1. slice trades_list with test date
         make prediction for each trade (based on regime data and rf_model). loop by strategy.
         prediction_results_df
    2. account = check_stop_loss_take_profit
        get_holdings_value_by_strategy
    3. buy_df = strategy_and_ticker_cash_allocation()
    4. buy_heap = create_buy_heap(buy_df)
       account = execute_sell_orders()
    5. account = execute_buy_orders() # Execute buy orders, create sl and tp prices
        update_account_portfolio_values


    """
    global train_tickers
    logger.info("Starting testing phase...")

    # Initialize testing variables
    strategy_to_coefficient = {}
    account = initialize_test_account()

    start_date = datetime.strptime(test_period_start, "%Y-%m-%d")
    end_date = datetime.strptime(test_period_end, "%Y-%m-%d")
    current_date = start_date
    account_values = pd.Series(index=pd.date_range(start=start_date, end=end_date))
    logger.info(f"Testing period: {start_date} to {end_date}")

    trades_list_db_name = os.path.join("PriceData", "trades_list_vectorised.db")
    con_tl = sqlite3.connect(trades_list_db_name)

    strategies_list = get_tables_list(con_tl, logger)
    # removes 'summary' table from list if it exists
    if "summary" in strategies_list:
        strategies_list.remove("summary")

    strategies_list = [strategies_list[0]]
    for strategy in strategies_list:
        """
        1. Generate predictions
        """
        # slice trades_list.db with test date
        query = f"SELECT * FROM strategy1 WHERE buy_date >= ? AND buy_date <= ?"
        trades_for_prediction_df = pd.read_sql(
            query, con_tl, params=(start_date, end_date)
        )
        logger.info(f"{trades_for_prediction_df}")

    # ----
    # Calculate final metrics and generate tear sheet
    metrics = calculate_metrics(account_values)
    wandb.log(metrics)

    logger.info("Final metrics calculated.")
    logger.info(metrics)

    try:
        generate_tear_sheet(account_values, filename=f"{benchmark_asset}_vs_strategy")
        logger.info("Tear sheet generated.")
    except Exception as e:
        logger.error(f"Error generating tear sheet: {e}")

    # Print final results
    logger.info("Testing Completed.")
    logger.info("-------------------------------------------------")
    logger.info(f"Account Cash: ${account['cash']: ,.2f}")
    logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
    logger.info("-------------------------------------------------")
    con_tl.close()


def get_holdings_value_by_strategy(account, ticker_price_history, current_date):
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
    prediction_results_df,
    account,
    holdings_value_by_strategy,
    prediction_threshold,
    asset_limit,
    strategy_limit,
):
    """
    Allocates cash to strategies and tickers based on prediction results and constraints.

    Parameters:
    - prediction_results_df (pd.DataFrame): DataFrame containing prediction results, including strategy names, tickers, actions, predictions, and accuracies.
    - account (dict): The current account state, including holdings, cash, and total portfolio value.
    - holdings_value_by_strategy (dict): Dictionary with the current value of holdings by strategy.
    - prediction_threshold (float): The minimum score required for a strategy to qualify.
    - asset_limit (float): The maximum proportion of the portfolio that can be allocated to a single asset.
    - strategy_limit (float): The maximum proportion of the portfolio that can be allocated to a single strategy.

    Returns:
    - pd.DataFrame: DataFrame containing the allocated cash for each strategy and ticker.
    """
    if prediction_results_df.empty:
        logger.info(
            "strategy_and_ticker_cash_allocation: prediction_results_df is empty."
        )
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = ["probability", "prediction", "action"]
    if not all(col in prediction_results_df.columns for col in required_cols):
        logger.error(
            f"strategy_and_ticker_cash_allocation: Missing required columns in prediction_results_df. Found: {prediction_results_df.columns}"
        )
        return pd.DataFrame()

    # Filter for actual buy signals confirmed by RF (prediction == 1)
    # Note: The loop appending to prediction_results_list already ensures action=='Buy' and prediction==1
    confirmed_buys_df = prediction_results_df[
        prediction_results_df["prediction"] == 1
    ].copy()

    if confirmed_buys_df.empty:
        logger.info(
            "strategy_and_ticker_cash_allocation: No confirmed buy signals (prediction == 1)."
        )
        return pd.DataFrame()

    # Use probability as the primary score for ranking
    confirmed_buys_df["score"] = confirmed_buys_df["probability"]

    # Sort by the new probability-based score
    confirmed_buys_df = confirmed_buys_df.sort_values(by=["score"], ascending=False)
    logger.info(
        f"strategy_and_ticker_cash_allocation: Sorted confirmed buys by probability score:\n{confirmed_buys_df[['strategy_name', 'ticker', 'probability', 'score']]}"
    )

    # Filter DataFrame where score (probability) > prediction_threshold
    qualifying_strategies_df = confirmed_buys_df[
        confirmed_buys_df["score"] > prediction_threshold
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    if qualifying_strategies_df.empty:
        logger.info(
            f"strategy_and_ticker_cash_allocation: No strategies qualify after probability threshold {prediction_threshold}."
        )
        return pd.DataFrame()

    logger.info(
        f"strategy_and_ticker_cash_allocation: Qualifying strategies after threshold:\n{qualifying_strategies_df[['strategy_name', 'ticker', 'score']]}"
    )

    total_portfolio_value = account["total_portfolio_value"]
    available_cash = account["cash"]

    max_strategy_investment = total_portfolio_value * strategy_limit
    print(f"{max_strategy_investment = }")

    qualifying_strategies_df["max_s_cash"] = max_strategy_investment

    # Adjust strategy allocation based on existing holdings
    holdings_value_df = pd.DataFrame.from_dict(
        holdings_value_by_strategy, orient="index", columns=["holdings_value"]
    )

    qualifying_strategies_df = qualifying_strategies_df.merge(
        holdings_value_df, left_on="strategy_name", right_index=True, how="left"
    )

    qualifying_strategies_df["holdings_value"] = qualifying_strategies_df[
        "holdings_value"
    ].fillna(0)
    qualifying_strategies_df["max_s_cash_adj"] = (
        qualifying_strategies_df["max_s_cash"]
        - qualifying_strategies_df["holdings_value"]
    )

    # Adjust by number_of_tickers_in_qualifying_strategies
    qualifying_strategies_df["ticker_count"] = qualifying_strategies_df.groupby(
        "strategy_name"
    )["ticker"].transform("count")

    # Max available cash for each strategy divided by the number of tickers in that strategy
    qualifying_strategies_df["max_s_t_cash"] = (
        qualifying_strategies_df["max_s_cash_adj"]
        / qualifying_strategies_df["ticker_count"]
    )

    # Calculate probability adjusted investment (using 'score' which is now probability)
    qualifying_strategies_df["prob_adj_investment"] = (
        qualifying_strategies_df["max_s_t_cash"]
        * qualifying_strategies_df["score"]  # Use score (probability) for weighting
    )

    # Calculate investment considering asset limit
    asset_limit_value = asset_limit * total_portfolio_value
    logger.info(f"strategy_and_ticker_cash_allocation: {asset_limit_value = }")

    # Allocate cash based on prob_adj_investment until cash is exhausted
    qualifying_strategies_df["allocated_cash"] = (
        0.0  # Ensure the column is of float type
    )

    for ticker, group in qualifying_strategies_df.groupby("ticker"):
        # Calculate the remaining cash available for this ticker after considering existing holdings
        existing_holding_value = sum(
            holding["quantity"] * group.iloc[0]["current_price"]
            for strategy, holding in account["holdings"].get(ticker, {}).items()
        )
        remaining_cash = min(
            max(0, asset_limit_value - existing_holding_value), available_cash
        )

        for index, row in group.iterrows():
            if remaining_cash <= 0:
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

    return qualifying_strategies_df[
        [
            "strategy_name",
            "ticker",
            "current_price",
            "score",
            "allocated_cash",
            "quantity",
        ]
    ][qualifying_strategies_df["allocated_cash"] > 0]


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


def update_account_portfolio_values(account, ticker_price_history, current_date):
    # Calculate and update total portfolio value
    total_value = account["cash"]
    for ticker, account_strategies in account["holdings"].items():
        for strategy, holding in account_strategies.items():
            current_price = ticker_price_history[ticker].loc[
                current_date.strftime("%Y-%m-%d")
            ]["Close"]
            total_value += holding["quantity"] * current_price
    account["total_portfolio_value"] = total_value

    return account


def generate_predictions(trades_for_prediction_df, strategy, logger):
    # load rf model
    if not check_model_exists(strategy):
        logger.error(f"Model for {strategy} does not exist, skipping...")
        return None
    rf_dict[strategy_name] = load_rf_model(strategy, logger)
    if rf_dict[strategy_name] is None:
        logger.error(f"Model for {strategy} could not be loaded, skipping...")
        return None
    assert isinstance(
        rf_dict[strategy_name], dict
    ), "loaded_model is not a dictionary, model loading failed."
    # logger.info(f"{rf_dict}")

    # make prediction
    sample_df = trades_for_prediction_df[["^VIX", "One_day_spy_return"]]
    # logger.info(f"{sample_df = }")
    trades_for_prediction_df["prediction"] = predict_random_forest_classifier(
        rf_dict["rf_classifier"], sample_df
    )

    # logger.info(f"{rf_dict["accuracy"] = }")
    trades_for_prediction_df["accuracy"] = round(rf_dict["accuracy"], 4)
    trades_for_prediction_df["strategy_name"] = strategy

    return trades_for_prediction_df


def insert_trade_into_tranding_account_db(
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
    try:
        with sqlite3.connect(trading_account_db_name) as conn:
            trades_df.to_sql(
                f"trades_{experiment_name}",
                conn,
                if_exists="append",
                index=True,
                dtype={"trade_id": "TEXT PRIMARY KEY"},
            )
    except Exception as e:
        logger.error(f"Error saving trades to database: {e}")


# needed?
def loop_strategies_get_predictions(strategies_list, trading_account_db_name):
    """ "
    not sure if needed any more
    """
    trades_with_prediction_all_startegies_df = pd.DataFrame()
    # strategies_list = [strategies_list[0]]
    for strategy in strategies_list:
        logger.info(f"{strategy}")
        """
        1. Generate predictions
        """
        # slice trades_list.db with test date
        with sqlite3.connect(trading_account_db_name) as conn:
            query = f"SELECT * FROM {strategy} WHERE buy_date >= ? AND buy_date <= ?"
            trades_for_prediction_df = pd.read_sql(
                query, conn, params=(start_date, end_date), index_col=["trade_id"]
            )
        # logger.info(f"{trades_for_prediction_df}")
        if trades_for_prediction_df.empty:
            logger.info(f"No trades found for {strategy} in the given date range.")
            continue

        trades_with_prediction_df = generate_predictions(
            trades_for_prediction_df, strategy, logger
        )
        # logger.info(f"DataFrame with predictions: {trades_with_prediction_df}")
        # logger.info(f"{trades_with_prediction_df.info() = }")

        trades_with_prediction_all_startegies_df = pd.concat(
            [
                trades_with_prediction_all_startegies_df.copy(),
                trades_with_prediction_df,
            ],
            axis=0,
        )

        # logger.info(f"DataFrame with predictions: {trades_with_prediction_df}")
        # logger.info(f"{trades_with_prediction_df.info() = }")
        if trades_with_prediction_df is not None:
            logger.info(f"{len(trades_with_prediction_df) = }")
        # logger.info(f"{len(trades_with_prediction_all_startegies_df) = }")

    logger.info(f"")
    logger.info(f"===SUMMARY===")
    logger.info(f"{len(trades_with_prediction_all_startegies_df) = }")

    trades_with_positive_prediction_df = trades_with_prediction_all_startegies_df[
        trades_with_prediction_all_startegies_df["prediction"] == 1
    ]
    logger.info(f"{trades_with_positive_prediction_df = }")
    logger.info(f"{len(trades_with_positive_prediction_df) = }")

    positive_prediction_and_threshold_df = trades_with_prediction_all_startegies_df[
        (trades_with_prediction_all_startegies_df["prediction"] == 1)
        & (trades_with_prediction_all_startegies_df["accuracy"] > accuracy_threshold)
    ]
    logger.info(f"{positive_prediction_and_threshold_df = }")
    logger.info(f"{len(positive_prediction_and_threshold_df) = }")

    # HACK: rename col Ticker to ticker
    trades_with_prediction_all_startegies_df.rename(
        columns={"Ticker": "ticker"}, inplace=True
    )

    with sqlite3.connect(trading_account_db_name) as conn:
        trades_with_prediction_all_startegies_df.to_sql(
            "trades_with_prediction_all_strategies",
            conn,
            if_exists="replace",
            index=True,
        )


if __name__ == "__main__":
    logger = setup_logging("logs", "testing.log", level=logging.INFO)

    # setup database connections
    strategy_decisions_final_db_name = os.path.join(
        "PriceData", "strategy_decisions_final.db"
    )
    trading_account_db_name = os.path.join("PriceData", "trading_account.db")
    # trades_list_db_name = os.path.join("PriceData", "trades_list_vectorised.db")
    # con_tl = sqlite3.connect(trades_list_db_name)
    # con_sd_final = sqlite3.connect(strategy_decisions_final_db_name)
    # con_trading_account = sqlite3.connect(trading_account_db_name)
    # con_pd = sqlite3.connect(PRICE_DB_PATH)

    # setup dates
    start_date = datetime.strptime(test_period_start, "%Y-%m-%d")
    test_period_end = "2025-01-06"
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
    strategies = strategies_top10_acc
    account = initialize_test_account()
    accuracy_threshold = 0.75
    prediction_threshold = 0.75
    use_rf_model_predictions = False
    # experiment_name = f"{use_rf_model_predictions = }_{len(train_tickers)}_{test_period_start}_{test_period_end}_{train_stop_loss}_{train_take_profit}_thres{prediction_threshold}"
    experiment_name = f"test"
    account_values = pd.Series(index=pd.date_range(start=start_date, end=end_date))
    rf_dict = {}

    # get price history for tickers
    ticker_price_history = {}
    try:
        with sqlite3.connect(PRICE_DB_PATH) as conn:
            for ticker in train_tickers + regime_tickers:
                query = f"SELECT * FROM '{ticker}' WHERE Date >= ? AND Date <= ?"
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

    for date in test_date_range:
        prediction_results_list = []
        date_missing = False
        for idx, strategy in enumerate(strategies):
            strategy_name = strategy.__name__
            logger.info(
                f"{date.strftime("%Y-%m-%d")} {strategy_name} {idx + 1}/{len(strategies)}"
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

            for ticker in train_tickers:
                try:
                    # Attempt to get the shifted decision
                    action = (
                        precomputed_decisions[strategy_name]
                        .shift(1)
                        .loc[date.strftime("%Y-%m-%d"), ticker]
                    )
                except KeyError:
                    # Log a warning and default action if the key is not found
                    logger.warning(
                        f"Precomputed decision not found for {ticker} on {date.strftime('%Y-%m-%d')} after shift. Defaulting to 'hold'."
                    )
                    action = "hold"

                try:
                    current_price = ticker_price_history[ticker].loc[
                        date.strftime("%Y-%m-%d")
                    ]["Close"]
                    current_price = round(current_price, 2)
                    # logger.info(
                    #     f"Current price: {ticker} {current_price} on {date.strftime('%Y-%m-%d')}."
                    # )
                except KeyError:
                    logger.warning(
                        f"Current price not found for {ticker} on {date.strftime('%Y-%m-%d')}."
                    )
                    continue

                # logger.info(
                #     f"{ticker} - {date.strftime("%Y-%m-%d")} - {action} - current_price: {current_price}"
                # )
                account = check_stop_loss_take_profit(
                    account, ticker, current_price, date, trading_account_db_name
                )
                # logger.info(f"{account = }")

                if action == "Buy":
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
                        prediction, probability = predict_random_forest_classifier(
                            rf_dict[strategy_name]["rf_classifier"],
                            sample_df,
                        )

                        if prediction != 1:
                            action = (
                                "hold"  # Override original 'Buy' if RF doesn't confirm
                            )

                        accuracy = round(
                            rf_dict[strategy_name]["accuracy"], 2
                        )  # Keep accuracy for logging/potential future use
                        probability = np.round(probability[0], 4)  # Round probability
                        logger.info(f"{probability = }")

                        logger.info(
                            f"Prediction {date.strftime('%Y-%m-%d')} {strategy_name} {ticker}: {prediction} (Prob: {probability:.4f}), Acc: {accuracy}, VIX: {daily_vix_df:.2f}, SPY: {One_day_spy_return:.2f}, Action: {action}"
                        )

                    # Only add to results if the original action was Buy and RF prediction is 1
                    if action == "Buy":

                        prediction_results_list.append(
                            {
                                "strategy_name": strategy_name,
                                "ticker": ticker,
                                "action": action,  # Should always be 'Buy' here
                                "prediction": prediction,  # Should always be 1 here
                                "accuracy": accuracy,  # Historical accuracy of the model
                                "probability": probability,  # Probability of this specific prediction being 1
                                "current_price": current_price,
                            }
                        )

                # execute sell orders
                if action == "sell":
                    account = execute_sell_orders(
                        action,
                        ticker,
                        strategy_name,
                        account,
                        current_price,
                        date,
                        logger,
                    )

        if not date_missing:

            # # Calculate holdings value by strategy. needed to ensure cash allocation constraints are met.
            holdings_value_by_strategy = get_holdings_value_by_strategy(
                account, ticker_price_history, date
            )
            logger.info(f"{holdings_value_by_strategy = }")

            prediction_results_df = pd.DataFrame(prediction_results_list)
            prediction_results_df["Date"] = date.strftime("%Y-%m-%d")
            prediction_results_df["prediction_id"] = (
                prediction_results_df["ticker"]
                + "_"
                + prediction_results_df["strategy_name"]
                + "_"
                + prediction_results_df["Date"]
            )
            prediction_results_df.set_index("prediction_id", inplace=True)
            # Ensure the DataFrame is not empty
            if not prediction_results_df.empty:
                logger.info(f"{len(prediction_results_df) = }")
                try:
                    with sqlite3.connect(trading_account_db_name) as conn:
                        prediction_results_df.to_sql(
                            f"predictions_{experiment_name}",
                            conn,
                            if_exists="replace",
                            index=True,
                            dtype={"trade_id": "TEXT PRIMARY KEY"},
                        )
                except Exception as e:
                    logger.error(f"Error saving predictions to database: {e}")
            else:
                logger.warning("No prediction_results_df to insert into the database.")

            buy_df = strategy_and_ticker_cash_allocation(
                prediction_results_df,
                account,
                holdings_value_by_strategy,
                prediction_threshold,
                train_trade_asset_limit,
                train_trade_strategy_limit,
            )

            buy_heap = create_buy_heap(buy_df)
            logger.info(f"{buy_heap = }")

            suggestion_heap = []
            # Execute buy orders, create sl and tp prices
            account = execute_buy_orders(
                buy_heap,
                suggestion_heap,
                account,
                date,
                train_trade_liquidity_limit,
                train_stop_loss,
                train_take_profit,
            )

            # Calculate and update account total portfolio value
            account = update_account_portfolio_values(
                account, ticker_price_history, date
            )
            # logger.info(f"{account = }")

            # Update account values for metrics
            total_value = account["total_portfolio_value"]
            account_values[date] = total_value
            # logger.info(f"{total_value = }")
            logger.info(f"total_portfolio_value: {round(total_value, 2)}")

    # Log final results
    logger.info("-------------------------------------------------")
    logger.info(f"Trades: {account['trades']}")
    # logger.info(f"Trades: {len(account['trades'])}")
    logger.info(f"Holdings: {account['holdings']}")
    logger.info(f"Account Cash: ${account['cash']: ,.2f}")
    logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
    # logger.info(f"Active Count: {active_count}")
    logger.info("-------------------------------------------------")

    try:
        # Calculate final metrics and generate tear sheet
        metrics = calculate_metrics(account_values)
        logger.info(metrics)
        generate_tear_sheet(
            account_values, filename=f"{benchmark_asset}_vs_strategy_{experiment_name}"
        )
        logger.info("Tear sheet generated.")
    except Exception as e:
        logger.error(f"Error generating tear sheet: {e}")
