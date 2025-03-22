import heapq
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import certifi
from numpy import int64
import pandas as pd
from pymongo import MongoClient

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from random_forest import predict_random_forest_classifier, train_and_store_classifiers

# from variables import config_dict
from TradeSim.variables import config_dict

import wandb
from config import FINANCIAL_PREP_API_KEY, mongo_url
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

train_tickers
from helper_files.client_helper import get_ndaq_tickers, store_dict_as_json, strategies
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


def check_stop_loss_take_profit(account, ticker, current_price):
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
                if (
                    current_price < holding["stop_loss"]
                    or current_price > holding["take_profit"]
                ):
                    account["trades"].append(
                        {
                            "symbol": ticker,
                            "quantity": holding["quantity"],
                            "price": current_price,
                            "action": "sell",
                            "strategy": strategy,
                        }
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
        print(f"Executing BUY order for {ticker} of quantity {quantity}")

        account["trades"].append(
            {
                "symbol": ticker,
                "quantity": quantity,
                "price": current_price,
                "action": "buy",
                "date": current_date.strftime("%Y-%m-%d"),
                "strategy": strategy_name,
            }
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
        account["trades"].append(
            {
                "symbol": ticker,
                "quantity": quantity,
                "price": round(current_price, 2),
                "action": "sell",
                "strategy": strategy_name,
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )
        account["cash"] += quantity * current_price
        del account["holdings"][ticker][strategy_name]
        logger.info(
            f"{ticker} - Sold {quantity} shares at ${current_price} for {strategy_name}"
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
                account = check_stop_loss_take_profit(account, ticker, current_price)

                # Get strategy decisions
                decisions_and_quantities = []
                portfolio_qty = account["holdings"].get(ticker, {}).get("quantity", 0)

                for strategy in strategies:
                    strategy_name = strategy.__name__

                    # Get precomputed strategy decision
                    # action = precomputed_decisions[strategy_name][ticker].get(date_str)

                    # if action is None:
                    #     # Skip if no precomputed decision (should not happen if properly precomputed)
                    #     logger.warning(
                    #         f"No precomputed decision for {ticker}, {strategy_name}, {date_str}"
                    #     )
                    #     continue

                    # Get precomputed strategy decision from DataFrame
                    action = precomputed_decisions[
                        (precomputed_decisions["Strategy"] == strategy_name)
                        & (precomputed_decisions["Ticker"] == ticker)
                        & (precomputed_decisions["Date"] == date_str)
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

                            prediction = predict_random_forest_classifier(
                                trained_classifiers[strategy_name]["rf_classifier"],
                                sample_df,
                            )

                            if prediction != 1:
                                action = "hold"
                            accuracy = trained_classifiers[strategy_name]["accuracy"]
                            logger.info(
                                f"Prediction {current_date} {strategy_name} {ticker}: {prediction}, {accuracy = } daily_vix_df = {daily_vix_df:.2f} {action = }"
                            )

                            weight = prediction * accuracy  # not needed

                        else:
                            prediction = None
                            accuracy = None
                            weight = 0

                        prediction_results_list.append(
                            {
                                "strategy_name": strategy_name,
                                "ticker": ticker,
                                "action": action,
                                "prediction": prediction,
                                "accuracy": accuracy,
                                "current_price": round(current_price, 2),
                            }
                        )

                        # prediction results for buy actions/signals store in dict
                        if strategy_name not in prediction_results:
                            prediction_results[strategy_name] = {}

                        prediction_results[strategy_name][ticker] = {
                            "action": action,
                            "prediction": prediction,
                            "accuracy": accuracy,
                        }

                    else:
                        weight = 0

                    # execute sell orders
                    if action == "sell":
                        account = execute_sell_orders(
                            action,
                            ticker,
                            strategy_name,
                            account,
                            current_price,
                            current_date,
                            logger,
                        )
                    # logger.info(f"{prediction_results = }")

                #     # Compute trade decision and quantity based on precomputed action and prediction
                #     decision, qty = compute_trade_quantities(
                #         action,
                #         current_price,
                #         account_cash,
                #         portfolio_qty,
                #         total_portfolio_value,
                #     )
                #     # decision, qty = compute_trade_quantities_only_buy_one(action, portfolio_qty)

                #     # weight = strategy_to_coefficient[strategy.__name__]
                #     decisions_and_quantities.append((decision, qty, weight))
                #     # logger.info(f"{ticker} {strategy_name} {decision = }, {qty = }, {weight = }")

                # logger.info(f"{decisions_and_quantities = }")
                # logger.info(f"{prediction_results = }")

                # # Process weighted decisions
                # (
                #     decision,
                #     quantity,
                #     buy_weight,
                #     sell_weight,
                #     hold_weight,
                # ) = weighted_majority_decision_and_median_quantity(
                #     decisions_and_quantities
                # )

                # logger.info(
                #     f"{ticker} - Decision: {decision}, Quantity: {quantity}, Buy Weight: {buy_weight}, Sell Weight: {sell_weight}, Hold Weight: {hold_weight}"
                # )

                # # Execute trading decisions
                # if (
                #     decision == "buy"
                #     and ((portfolio_qty + quantity) * current_price)
                #     / account["total_portfolio_value"]
                #     <= train_trade_asset_limit
                # ):
                #     heapq.heappush(
                #         buy_heap,
                #         (
                #             -(buy_weight - (sell_weight + (hold_weight * 0.5))),
                #             quantity,
                #             ticker,
                #         ),
                #     )

                # elif decision == "sell" and ticker in account["holdings"]:
                #     quantity = max(quantity, 1)
                #     quantity = account["holdings"][ticker]["quantity"]
                #     account["trades"].append(
                #         {
                #             "symbol": ticker,
                #             "quantity": quantity,
                #             "price": current_price,
                #             "action": "sell",
                #             "date": current_date.strftime("%Y-%m-%d"),
                #         }
                #     )
                #     account["cash"] += quantity * current_price
                #     del account["holdings"][ticker]
                #     logger.info(
                #         f"{ticker} - Sold {quantity} shares at ${current_price}"
                #     )

                # elif (
                #     portfolio_qty == 0.0
                #     and buy_weight > sell_weight
                #     and ((quantity * current_price) / account["total_portfolio_value"])
                #     < trade_asset_limit
                #     and float(account["cash"]) >= train_trade_liquidity_limit
                # ):
                #     max_investment = (
                #         account["total_portfolio_value"] * train_trade_asset_limit
                #     )
                #     buy_quantity = min(
                #         int(max_investment // current_price),
                #         int(account["cash"] // current_price),
                #     )
                #     if buy_weight > train_suggestion_heap_limit:
                #         buy_quantity = max(2, buy_quantity)
                #         buy_quantity = buy_quantity // 2
                #         heapq.heappush(
                #             suggestion_heap,
                #             (-(buy_weight - sell_weight), buy_quantity, ticker),
                #         )

        # Convert list of dictionaries to DataFrame
        prediction_results_df = pd.DataFrame(prediction_results_list)
        # Save DataFrame to CSV in results folder
        csv_file_path = os.path.join(results_dir, "prediction_results.csv")
        prediction_results_df.to_csv(csv_file_path, index=False)

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
        # Update portfolio values
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, strategies, trading_simulator, ticker_price_history, logger
        )

        # Update time delta needed?
        # time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # # Calculate and update total portfolio value
        total_value = account["cash"]
        for ticker, account_strategies in account["holdings"].items():
            for strategy, holding in account_strategies.items():
                current_price = ticker_price_history[ticker].loc[
                    current_date.strftime("%Y-%m-%d")
                ]["Close"]
                total_value += holding["quantity"] * current_price
        account["total_portfolio_value"] = total_value

        # Calculate and update total portfolio value
        # total_value = account["cash"]
        # for ticker in account["holdings"]:
        #     current_price = ticker_price_history[ticker].loc[
        #         current_date.strftime("%Y-%m-%d")
        #     ]["Close"]
        #     total_value += account["holdings"][ticker]["quantity"] * current_price
        # account["total_portfolio_value"] = total_value

        # Update account values for metrics
        account_values[current_date] = total_value

        # Update rankings #needed?
        rank = update_strategy_ranks(strategies, points, trading_simulator)

        # Log daily results
        logger.info("-------------------------------------------------")
        logger.info(f"Account Cash: ${account['cash']: ,.2f}")
        logger.info(f"Trades: {account['trades']}")
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
    prediction_results_df["score"] = (
        prediction_results_df["accuracy"] * prediction_results_df["prediction"]
    )

    prediction_results_df.sort_values(by=["score"], ascending=False)

    # Filter DataFrame where score > prediction_threshold
    qualifying_strategies_df = prediction_results_df[
        prediction_results_df["score"] > prediction_threshold
    ]

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

    # Calculate accuracy adjusted investment
    qualifying_strategies_df["accuracy_adj_investment"] = (
        qualifying_strategies_df["max_s_t_cash"] * qualifying_strategies_df["accuracy"]
    )

    # Calculate investment considering asset limit
    asset_limit_value = asset_limit * total_portfolio_value
    print(f"{asset_limit_value = }")

    # Allocate cash based on accuracy_adj_investment until cash is exhausted
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
            allocation = min(row["accuracy_adj_investment"], remaining_cash)
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


if __name__ == "__main__":
    # account = initialize_test_account()
    current_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
    account = {
        "holdings": {
            # "AAPL": {
            #     "quantity": 10,
            #     "price": 100,
            #     "strategy": "MEDPRICE_indicator",
            # },
            "AAPL": {
                "MEDPRICE_indicator": {
                    "quantity": 10,
                    "price": 100,
                },
                "TYPPRICE_indicator": {
                    "quantity": 1,
                    "price": 50,
                },
            },
            "MSFT": {
                "WCLPRICE_indicator": {
                    "quantity": 10,
                    "price": 300,
                },
            },
        },
        "cash": train_start_cash,
        # "cash": 500,
        "trades": [],
        "total_portfolio_value": train_start_cash,
    }

    holdings_value_by_strategy = {
        "MEDPRICE_indicator": 1500.0,
        "TYPPRICE_indicator": 500.0,
        "WCLPRICE_indicator": 3000.0,
    }

    # Load DataFrame from CSV in results folder
    csv_file_path = os.path.join(results_dir, "prediction_results2.csv")
    prediction_results_df = pd.read_csv(csv_file_path)

    prediction_threshold = 0.5
    # asset_limit = train_trade_asset_limit
    asset_limit = 0.25
    strategy_limit = train_trade_strategy_limit
    train_trade_liquidity_limit = 10

    buy_df = strategy_and_ticker_cash_allocation(
        prediction_results_df,
        account,
        holdings_value_by_strategy,
        prediction_threshold,
        asset_limit,
        strategy_limit,
    )
    print(buy_df)

    buy_heap = create_buy_heap(buy_df)

    print(buy_heap)

    suggestion_heap = []

    account = execute_buy_orders(
        buy_heap,
        suggestion_heap,
        account,
        current_date,
        train_trade_liquidity_limit,
        train_stop_loss,
        train_take_profit,
    )

    print(account)

    # Create ticker_price_history_df
    # Example data for AAPL
    data_aapl = {
        "Date": ["2025-03-20", "2025-03-21"],
        "Open": [148.00, 150.00],
        "High": [151.00, 153.00],
        "Low": [147.00, 149.00],
        "Close": [150.00, 152.00],
        "Volume": [1000000, 1100000],
    }

    # Example data for MSFT
    data_msft = {
        "Date": ["2025-03-20", "2025-03-21"],
        "Open": [248.00, 250.00],
        "High": [251.00, 255.00],
        "Low": [247.00, 249.00],
        "Close": [250.00, 255.00],
        "Volume": [2000000, 2100000],
    }

    # Convert to DataFrame and set the Date as the index
    df_aapl = pd.DataFrame(data_aapl)
    df_aapl["Date"] = pd.to_datetime(df_aapl["Date"])
    df_aapl.set_index("Date", inplace=True)

    df_msft = pd.DataFrame(data_msft)
    df_msft["Date"] = pd.to_datetime(df_msft["Date"])
    df_msft.set_index("Date", inplace=True)

    # Create ticker_price_history dictionary
    ticker_price_history = {
        "AAPL": df_aapl,
        "MSFT": df_msft,
    }

    # account = update_account_portfolio_values(
    #     account, ticker_price_history, current_date
    # )
    # print(account)

    account = {
        "holdings": {
            "AAPL": {
                "MEDPRICE_indicator": {
                    "quantity": 45.06,
                    "price": 242.4334412,
                    "stop_loss": 235.160437964,
                    "take_profit": 254.55511326,
                },
                "TYPPRICE_indicator": {
                    "quantity": 6.5,
                    "price": 242.4334412,
                    "stop_loss": 235.160437964,
                    "take_profit": 254.55511326,
                },
            },
            "MSFT": {
                "WCLPRICE_indicator": {
                    "quantity": 18.259999999999998,
                    "price": 423.7104187,
                    "stop_loss": 410.999106139,
                    "take_profit": 444.895939635,
                },
                "TYPPRICE_indicator": {
                    "quantity": 11.21,
                    "price": 423.7104187,
                    "stop_loss": 410.999106139,
                    "take_profit": 444.895939635,
                },
            },
        },
        "cash": 31917.257772839002,
        "trades": [
            {
                "symbol": "MSFT",
                "quantity": 8.26,
                "price": 423.7104187,
                "action": "buy",
                "date": "2021-01-01",
                "strategy": "WCLPRICE_indicator",
            },
            {
                "symbol": "MSFT",
                "quantity": 11.21,
                "price": 423.7104187,
                "action": "buy",
                "date": "2021-01-01",
                "strategy": "TYPPRICE_indicator",
            },
            {
                "symbol": "AAPL",
                "quantity": 35.06,
                "price": 242.4334412,
                "action": "buy",
                "date": "2021-01-01",
                "strategy": "MEDPRICE_indicator",
            },
            {
                "symbol": "AAPL",
                "quantity": 5.5,
                "price": 242.4334412,
                "action": "buy",
                "date": "2021-01-01",
                "strategy": "TYPPRICE_indicator",
            },
        ],
        "total_portfolio_value": 50000.0,
    }

    Holdings = {
        "MSFT": {
            "STOCHRSI_indicator": {
                "quantity": 11.97,
                "price": 417.74,
                "stop_loss": 405.2078,
                "take_profit": 438.627,
            }
        },
        "AAPL": {
            "STOCHRSI_indicator": {
                "quantity": 20.53,
                "price": 243.58,
                "stop_loss": 236.2726,
                "take_profit": 255.75900000000001,
            },
            "HT_PHASOR_indicator": {
                "quantity": 0.24,
                "price": 243.09,
                "stop_loss": 235.7973,
                "take_profit": 255.24450000000002,
            },
        },
    }

    Trades = [
        {
            "symbol": "MSFT",
            "quantity": 11.97,
            "price": 417.74,
            "action": "buy",
            "date": "2025-01-02",
            "strategy": "STOCHRSI_indicator",
        },
        {
            "symbol": "AAPL",
            "quantity": 20.53,
            "price": 243.58,
            "action": "buy",
            "date": "2025-01-02",
            "strategy": "STOCHRSI_indicator",
        },
        {
            "symbol": "AAPL",
            "quantity": 0.24,
            "price": 243.09,
            "action": "buy",
            "date": "2025-01-03",
            "strategy": "HT_PHASOR_indicator",
        },
    ]
    # account = execute_buy_orders(
    #     buy_heap, suggestion_heap, account, ticker_price_history, current_date
    # )

    # todo:
    # tests
    # need to update functions to handle modified account structure
    # sell_df

    # Execute buy orders
    # buy_heap = []
    # for index, row in qualifing_strategies_df.iterrows():
    #     if row["allocated_cash"] > 0:
    #         heapq.heappush(
    #             buy_heap,
    #             (
    #                 -row["score"],
    #                 row["allocated_cash"] // row["current_price"],
    #                 row["ticker"],
    #                 row["strategy_name"],
    #             ),
    #         )

    # account = execute_buy_orders(
    #     buy_heap, [], account, ticker_price_history, current_date
    # )
