from datetime import datetime
import re
import sqlite3
import time
import pandas as pd
import os, sys
import logging
import numpy as np
from typing import Literal
from sklearn.metrics import accuracy_score

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from random_forest import (
    predict_random_forest_classifier,
    train_random_forest_classifier,
)
from helper_files.client_helper import strategies, strategies_test
from helper_files.client_helper import setup_logging

from config import PRICE_DB_PATH


def walk_forward_analysis(
    trades: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    in_sample_period: int,
    out_sample_period: int,
    strategy_name: str,
    required_features: list[str],
) -> pd.DataFrame:
    current_date = start_date
    all_results = []

    while (
        current_date + pd.Timedelta(days=in_sample_period + out_sample_period)
        <= end_date
    ):
        # Define in-sample and out-of-sample windows
        # in_sample_mask = (trades["buy_date"] >= current_date) & (
        in_sample_mask = (trades["buy_date"] >= start_date) & (
            trades["buy_date"] < current_date + pd.Timedelta(days=in_sample_period)
        )
        out_sample_mask = (
            trades["buy_date"] >= current_date + pd.Timedelta(days=in_sample_period)
        ) & (
            trades["buy_date"]
            < current_date + pd.Timedelta(days=in_sample_period + out_sample_period)
        )

        in_sample_trades = trades.loc[in_sample_mask]
        out_sample_trades = trades.loc[out_sample_mask]

        if in_sample_trades.empty or out_sample_trades.empty:
            logger.warning(
                f"Skipping window starting {current_date} due to empty in-sample or out-sample data."
            )
            current_date += pd.Timedelta(days=out_sample_period)
            continue

        # Train Random Forest Classifier
        try:
            rf_classifier, train_accuracy, precision, recall = (
                train_random_forest_classifier(in_sample_trades)
            )
        except Exception as train_error:
            logger.error(
                f"Error training model for window starting {current_date}: {train_error}"
            )
            current_date += pd.Timedelta(days=out_sample_period)
            continue

        # Prepare features from OUT-OF-SAMPLE data for prediction
        # required_features = ["^VIX", "One_day_spy_return"]
        if not all(col in out_sample_trades.columns for col in required_features):
            logger.error(
                f"Missing required feature columns in out-of-sample data for window starting {current_date}. Skipping."
            )
            current_date += pd.Timedelta(days=out_sample_period)
            continue
        out_sample_features = out_sample_trades[required_features]

        # Predict on out-of-sample data
        try:
            prediction, probability = predict_random_forest_classifier(
                rf_classifier, out_sample_features
            )
        except Exception as predict_error:
            logger.error(
                f"Error predicting for window starting {current_date}: {predict_error}"
            )
            current_date += pd.Timedelta(days=out_sample_period)
            continue

        # Ensure prediction is array-like before checking length
        if not isinstance(prediction, (list, np.ndarray)):
            prediction = np.array([prediction])  # Convert scalar to array

        # Create results for this window
        if len(prediction) != len(out_sample_trades):
            logger.error(
                f"Prediction length ({len(prediction)}) mismatch with out-of-sample data ({len(out_sample_trades)}) for window starting {current_date}. Skipping."
            )
            current_date += pd.Timedelta(days=out_sample_period)
            continue

        # Log the in-sample and out-of-sample periods
        in_sample_start_str = in_sample_trades["buy_date"].min().strftime("%Y-%m-%d")
        in_sample_end_str = in_sample_trades["buy_date"].max().strftime("%Y-%m-%d")
        out_sample_start_str = out_sample_trades["buy_date"].min().strftime("%Y-%m-%d")
        out_sample_end_str = out_sample_trades["buy_date"].max().strftime("%Y-%m-%d")

        current_accuracy = None
        try:
            current_accuracy = accuracy_score(
                out_sample_trades["returnB"],
                prediction,
            )

            logger.debug(
                f"{strategy_name} in: {in_sample_start_str}-{in_sample_end_str} out: {out_sample_start_str}-{out_sample_end_str} Accuracy: {current_accuracy:.4f}"
            )

        except Exception as acc_error:
            logger.warning(
                f"Could not calculate accuracy for window starting {current_date}: {acc_error}"
            )

        results_df = pd.DataFrame(
            {
                "date": out_sample_trades["buy_date"],
                "trade_id": out_sample_trades["trade_id"],
                "strategy": strategy_name,
                "ratio": out_sample_trades["ratio"],
                "predicted_outcome": prediction,
                "probability": np.round(probability, 3),
                "train_accuracy": np.round(train_accuracy, 3),
                "precision": np.round(precision, 3),
                "recall": np.round(recall, 3),
                "in_sample_start": in_sample_start_str,
                "in_sample_end": in_sample_end_str,
                "out_sample_start": out_sample_start_str,
                "out_sample_end": out_sample_end_str,
                "actual_outcome": out_sample_trades["returnB"],
                "current_accuracy": (
                    np.round(current_accuracy, 3)
                    if current_accuracy is not None
                    else None
                ),
            },
            index=out_sample_trades.index,
        )

        all_results.append(results_df)

        current_date += pd.Timedelta(days=out_sample_period)

    if all_results:
        final_results_df = pd.concat(all_results)
    else:
        logger.warning(f"No results generated for strategy {strategy_name}.")
        return pd.DataFrame()

    return final_results_df


def calculate_overall_metrics(results_df: pd.DataFrame) -> float | None:
    """Calculates overall accuracy from the results DataFrame."""
    if (
        results_df.empty
        or "actual_outcome" not in results_df.columns
        or "predicted_outcome" not in results_df.columns
    ):
        logger.warning(
            "Cannot calculate metrics: DataFrame is empty or missing required columns."
        )
        return None
    try:
        accuracy = accuracy_score(
            results_df["actual_outcome"],
            results_df["predicted_outcome"],
        )
        return float(accuracy)  # Cast to standard Python float
    except Exception as e:
        return None


def save_results_to_db(
    df: pd.DataFrame,
    table_name: str,
    conn: sqlite3.Connection,
    index: bool = True,
    index_label: str | None = None,
    if_exists: Literal["fail", "replace", "append"] = "replace",
):
    """Saves a DataFrame to a specified table in the SQLite database."""
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping save to table '{table_name}'.")
        return False
    try:
        df.to_sql(
            table_name, conn, if_exists=if_exists, index=index, index_label=index_label
        )
        logger.info(f"Successfully wrote {len(df)} records to table '{table_name}'.")
        return True
    except Exception as db_error:
        logger.error(f"Error writing to table '{table_name}': {db_error}")
        return False


def prepare_sp_return_data():
    """
    load ^GSPC historical data from price_data.db

    """
    with sqlite3.connect(PRICE_DB_PATH) as conn:
        query = """
            SELECT * FROM '^GSPC'
        """
        df = pd.read_sql(query, conn)

    # Ensure 'Date' is datetime and normalize it
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.set_index("Date", inplace=True)

    periods_list = [1, 5, 10, 20, 30, 60, 90, 120, 180, 365]
    for period in periods_list:
        df[f"spy_return({period})"] = df["Close"].pct_change(periods=period)
    # Calculate daily returns
    # df["One_day_spy_return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    # Save to SQLite database
    # with sqlite3.connect(price_data_db_name) as conn:
    #     df.to_sql("spy_returns", conn, if_exists="replace", index=False)
    return df


def main(strategies):
    """
    Step 1: Set Up Database Connections
    """
    # Define database paths
    price_data_dir = "PriceData"
    trades_list_db_name = os.path.join(price_data_dir, "trades_list_vectorised.db")
    results_db_name = os.path.join(price_data_dir, "backtest_results.db")
    required_features = ["^VIX", "One_day_spy_return"]

    strategies = [strategies[1]]

    # Use with statements for automatic connection management
    with sqlite3.connect(trades_list_db_name) as trades_conn, sqlite3.connect(
        results_db_name
    ) as results_conn:

        for idx, strategy in enumerate(strategies):
            start_time = time.time()
            strategy_name = strategy.__name__
            logger.info(f"{strategy_name} ({idx + 1}/{len(strategies)})")
            # Load trades data
            trades_df = pd.DataFrame()
            try:
                trades_df = pd.read_sql(f"SELECT * FROM {strategy_name}", trades_conn)
            except Exception as e:
                logger.error(
                    f"Error loading trades data for {strategy_name}, skipping: {e}"
                )
                continue
            trades_df["returnB"] = np.where(trades_df["ratio"] > 1, 1, 0)

            # make sure trade_df['buy_date'] is in datetime format and normalize
            trades_df["buy_date"] = pd.to_datetime(trades_df["buy_date"]).dt.normalize()
            trades_df["sell_date"] = pd.to_datetime(
                trades_df["sell_date"]
            ).dt.normalize()

            """
            Step 2: Define Walk-Forward Parameters
            """
            # Define in-sample and out-of-sample periods (in days)
            in_sample_period = 365 * 10
            out_sample_period = 30
            start_date = trades_df["buy_date"].min()
            end_date = trades_df["buy_date"].max()

            """
            step 3: Join regime data to trades_df
            """
            spy_df = prepare_sp_return_data()
            # join spy_df to trades_df on buy_date
            trades_df = trades_df.join(
                # spy_df[["One_day_spy_return"]], on="buy_date", how="left"
                spy_df[
                    [
                        "spy_return(1)",
                        "spy_return(5)",
                        "spy_return(10)",
                        "spy_return(20)",
                        "spy_return(30)",
                        "spy_return(60)",
                        "spy_return(90)",
                        "spy_return(120)",
                        "spy_return(180)",
                        "spy_return(365)",
                    ]
                ],
                on="buy_date",
                how="left",
            )

            # print(trades_df.columns)
            # print(trades_df)
            # return

            required_features = [
                "^VIX",
                "spy_return(1)",
                "spy_return(5)",
                "spy_return(10)",
                "spy_return(20)",
                "spy_return(30)",
                "spy_return(60)",
                "spy_return(90)",
                "spy_return(120)",
                "spy_return(180)",
                "spy_return(365)",
            ]

            missing_features = [
                f for f in required_features if f not in trades_df.columns
            ]
            if missing_features:
                logger.error(
                    f"Missing required features after join for {strategy_name}: {missing_features}. Skipping."
                )
                continue

            """
            Step 4: Walk-Forward Logic
            """
            # Call the modified walk_forward_analysis which now returns the DataFrame
            walk_forward_results = walk_forward_analysis(
                trades_df,
                start_date,
                end_date,
                in_sample_period,
                out_sample_period,
                strategy_name,
                required_features,
            )

            """
            Step 5: Analyze Results and Save
            """
            if not walk_forward_results.empty:
                # Prepare DataFrame for saving (set index)
                if "trade_id" in walk_forward_results.columns:
                    results_to_save = walk_forward_results.set_index("trade_id")
                    index_label = "trade_id"
                    use_index = True
                else:
                    logger.warning(
                        f"'trade_id' column not found in results for {strategy_name}. Writing without index."
                    )
                    results_to_save = walk_forward_results
                    index_label = None
                    use_index = False

                save_successful = save_results_to_db(
                    results_to_save,
                    strategy_name,
                    results_conn,
                    index=use_index,
                    index_label=index_label,
                    if_exists="replace",
                )

                if save_successful:
                    overall_accuracy = calculate_overall_metrics(walk_forward_results)

                    if overall_accuracy is not None:
                        logger.info(
                            f"{strategy_name} Overall Walk-Forward Accuracy: {overall_accuracy:.4f}"
                        )
                        overall_results_df = pd.DataFrame(
                            {
                                "strategy": [strategy_name],
                                "accuracy": [np.round(overall_accuracy, 4)],
                                "required_features": [str(required_features)],
                                # add regime data
                            }
                        )
                        save_results_to_db(
                            overall_results_df,
                            "overall_results",
                            results_conn,
                            index=False,
                            if_exists="append",
                        )
                    else:
                        logger.error(
                            f"Could not calculate overall accuracy for {strategy_name}."
                        )
                else:
                    logger.error(
                        f"Skipping overall analysis for {strategy_name} due to failure saving walk-forward results."
                    )

            else:
                logger.warning(
                    f"Skipping results analysis and saving for {strategy_name} due to empty walk_forward_results."
                )

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(
                f"Execution time for {strategy_name}: {elapsed_time:.2f} seconds"
            )

    logger.info("Database connections automatically closed.")


if __name__ == "__main__":
    logger = setup_logging("logs", "walk_forward.log", level=logging.INFO)
    main(strategies)
