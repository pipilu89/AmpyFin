from datetime import datetime
import sqlite3
import time
import pandas as pd
import os, sys
import logging
import numpy as np
from sklearn.metrics import accuracy_score

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from random_forest import (
    predict_random_forest_classifier,
    train_random_forest_classifier,
)
from helper_files.client_helper import strategies, strategies_test
from helper_files.client_helper import setup_logging


def walk_forward_analysis(
    trades,
    start_date,
    end_date,
    in_sample_period,
    out_sample_period,
    strategy_name,
    results_conn,
):
    current_date = start_date
    all_results = []

    while (
        current_date + pd.Timedelta(days=in_sample_period + out_sample_period)
        <= end_date
    ):
        # Define in-sample and out-of-sample windows
        # in_sample_mask = (trades["buy_date"] >= start_date) & (
        in_sample_mask = (trades["buy_date"] >= current_date) & (
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
        required_features = ["^VIX", "One_day_spy_return"]
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
            },
            index=out_sample_trades.index,
        )

        all_results.append(results_df)

        try:
            current_accuracy = accuracy_score(
                results_df["actual_outcome"], results_df["predicted_outcome"]
            )

            logger.info(
                f"{strategy_name} in: {in_sample_start_str}-{in_sample_end_str} out: {out_sample_start_str}-{out_sample_end_str} Accuracy: {current_accuracy:.4f}"
            )

        except Exception as acc_error:
            logger.warning(
                f"Could not calculate accuracy for window starting {current_date}: {acc_error}"
            )

        current_date += pd.Timedelta(days=out_sample_period)

    if all_results:
        final_results_df = pd.concat(all_results)
        if "trade_id" in final_results_df.columns:
            final_results_df.set_index("trade_id", inplace=True)
        else:
            logger.warning(
                f"'trade_id' column not found in results for {strategy_name}. Writing without index."
            )

        try:
            final_results_df.to_sql(
                f"{strategy_name}", results_conn, if_exists="replace", index=True
            )
            logger.info(
                f"Successfully wrote {len(final_results_df)} results for {strategy_name} to DB."
            )
        except Exception as db_error:
            logger.error(f"Error writing results to DB for {strategy_name}: {db_error}")
            return pd.DataFrame()
    else:
        logger.warning(f"No results generated for strategy {strategy_name}.")
        return pd.DataFrame()

    return final_results_df


def main(strategies):
    """
    Step 1: Set Up Database Connections
    """
    # Connect to SQLite databases
    price_data_dir = "PriceData"
    trades_list_db_name = os.path.join(price_data_dir, "trades_list_vectorised.db")
    trades_conn = sqlite3.connect(trades_list_db_name)
    results_db_name = os.path.join(price_data_dir, "backtest_results.db")
    results_conn = sqlite3.connect(results_db_name)

    strategies = [strategies[1], strategies[2]]
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

        # make sure trade_df['buy_date'] is in datetime format
        trades_df["buy_date"] = pd.to_datetime(trades_df["buy_date"])
        trades_df["sell_date"] = pd.to_datetime(trades_df["sell_date"])

        """
        Step 2: Define Walk-Forward Parameters
        """
        # Define in-sample and out-of-sample periods (in days)
        in_sample_period = 251 * 15
        out_sample_period = 22
        start_date = trades_df["buy_date"].min()
        end_date = trades_df["buy_date"].max()

        """
        Step 4: Walk-Forward Logic
        """

        walk_forward_results = walk_forward_analysis(
            trades_df,
            start_date,
            end_date,
            in_sample_period,
            out_sample_period,
            strategy_name,
            results_conn,
        )
        """
        Step 5: Analyze Results
        """
        if not walk_forward_results.empty:
            try:
                overall_accuracy = accuracy_score(
                    walk_forward_results["actual_outcome"],
                    walk_forward_results["predicted_outcome"],
                )
                logger.info(
                    f"{strategy_name} Overall Walk-Forward Accuracy: {overall_accuracy:.4f}"
                )

                overall_results_df = pd.DataFrame(
                    {
                        "strategy": [strategy_name],
                        "accuracy": [overall_accuracy],
                    }
                )
                try:
                    overall_results_df.to_sql(
                        "overall_results",
                        results_conn,
                        if_exists="append",
                        index=False,
                    )
                except Exception as db_error:
                    logger.error(
                        f"Error writing overall results to DB for {strategy_name}: {db_error}"
                    )
            except Exception as analysis_error:
                logger.error(
                    f"Error during overall analysis for {strategy_name}: {analysis_error}"
                )
        else:
            logger.warning(
                f"Skipping overall analysis for {strategy_name} due to no results generated or DB write failure."
            )
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        logger.info(f"Execution time for strategy: {elapsed_time:.2f} seconds")

    trades_conn.close()
    results_conn.close()
    logger.info("Database connections closed.")


if __name__ == "__main__":
    logger = setup_logging("logs", "walk_forward.log", level=logging.INFO)
    main(strategies)
