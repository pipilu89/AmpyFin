import heapq
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import certifi
import pandas as pd
from pymongo import MongoClient

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from random_forest import predict_random_forest_classifier, train_and_store_classifiers
from variables import config_dict

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
    train_trade_asset_limit,
    train_trade_liquidity_limit,
    regime_tickers,
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
    Checks and executes stop loss and take profit orders
    """
    if ticker in account["holdings"]:
        if account["holdings"][ticker]["quantity"] > 0:
            if (
                current_price < account["holdings"][ticker]["stop_loss"]
                or current_price > account["holdings"][ticker]["take_profit"]
            ):
                account["trades"].append(
                    {
                        "symbol": ticker,
                        "quantity": account["holdings"][ticker]["quantity"],
                        "price": current_price,
                        "action": "sell",
                    }
                )
                account["cash"] += (
                    account["holdings"][ticker]["quantity"] * current_price
                )
                del account["holdings"][ticker]
    return account


def execute_buy_orders(
    buy_heap, suggestion_heap, account, ticker_price_history, current_date
):
    """
    Executes buy orders from the buy and suggestion heaps
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

        _, quantity, ticker = heapq.heappop(heap)
        # print(f"Executing BUY order for {ticker} of quantity {quantity}")
        current_price = ticker_price_history[ticker].loc[
            current_date.strftime("%Y-%m-%d")
        ]["Close"]

        account["trades"].append(
            {
                "symbol": ticker,
                "quantity": quantity,
                "price": current_price,
                "action": "buy",
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

        account["cash"] -= quantity * current_price
        account["holdings"][ticker] = {
            "quantity": quantity,
            "price": current_price,
            "stop_loss": current_price * (1 - train_stop_loss),
            "take_profit": current_price * (1 + train_take_profit),
        }

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
    ticker_price_history, ideal_period, mongo_client, precomputed_decisions, logger
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

    # Get rank coefficients from database
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
                    action = precomputed_decisions[strategy_name][ticker].get(date_str)

                    if action is None:
                        # Skip if no precomputed decision (should not happen if properly precomputed)
                        logger.warning(
                            f"No precomputed decision for {ticker}, {strategy_name}, {date_str}"
                        )
                        continue

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

        # Execute buy orders
        account = execute_buy_orders(
            buy_heap, suggestion_heap, account, ticker_price_history, current_date
        )

        # Simulate ranking updates
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

        # Update time delta
        # time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # Calculate and update total portfolio value
        total_value = account["cash"]
        for ticker in account["holdings"]:
            current_price = ticker_price_history[ticker].loc[
                current_date.strftime("%Y-%m-%d")
            ]["Close"]
            total_value += account["holdings"][ticker]["quantity"] * current_price
        account["total_portfolio_value"] = total_value

        # Update account values for metrics
        account_values[current_date] = total_value

        # Update rankings
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


def strategy_weights(prediction_results_df):
    """
    Calculates the weights for each strategy based on the prediction results
    """
    high_threshold_bonus = 100
    low_threshold_bonus = 1
    strategy_weights = {}
    for strategy_name in prediction_results_df["strategy_name"].unique():
        strategy_df = prediction_results_df[
            prediction_results_df["strategy_name"] == strategy_name
        ]
        strategy_df = strategy_df.dropna()
        strategy_df = strategy_df[strategy_df["action"] == "Buy"]
        strategy_df = strategy_df[strategy_df["prediction"] == 1]
        strategy_df = strategy_df[strategy_df["accuracy"] > 0.75]

        strategy_weights[strategy_name] = len(strategy_df) * high_threshold_bonus

    for strategy_name in prediction_results_df["strategy_name"].unique():
        strategy_df = prediction_results_df[
            prediction_results_df["strategy_name"] == strategy_name
        ]
        strategy_df = strategy_df.dropna()
        strategy_df = strategy_df[strategy_df["action"] == "Buy"]
        strategy_df = strategy_df[strategy_df["prediction"] == 1]
        strategy_df = strategy_df[strategy_df["accuracy"] > 0.51]

        strategy_weights[strategy_name] += len(strategy_df) * low_threshold_bonus

    return strategy_weights


def strategy_and_tickers_weights(prediction_results_df):
    prediction_results_df["score"] = (
        prediction_results_df["accuracy"] * prediction_results_df["prediction"]
    )

    # return prediction_results_df.groupby(["strategy_name", "ticker"])["score"].sum()
    return (
        prediction_results_df.groupby(["strategy_name", "ticker"])
        .sum()
        .sort_values(by=["score", "strategy_name"], ascending=[False, True])
    )


if __name__ == "__main__":
    # Load DataFrame from CSV in results folder
    csv_file_path = os.path.join(results_dir, "prediction_results.csv")
    prediction_results_df = pd.read_csv(csv_file_path)

    # strategy_weights_df = strategy_weights(prediction_results_df)
    strategy_weights_df = strategy_and_tickers_weights(prediction_results_df)
    print(strategy_weights_df)
