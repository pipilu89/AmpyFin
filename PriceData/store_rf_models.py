import logging
import os
import sys
import sqlite3
import pandas as pd
import pickle
import time
from pydantic import NonNegativeInt
from scipy.stats import randint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from random_forest import (
    predict_random_forest_classifier,
    train_random_forest_classifier_RandomizedSearchCV,
    train_random_forest_classifier,
)


from helper_files.client_helper import setup_logging, strategies_test, strategies

from control import (
    test_period_end,
    train_period_start,
    train_period_end,
    train_tickers,
    regime_tickers,
    train_time_delta,
)


def store_rf_model_to_disk(rf_classifier, strategy_name, price_data_dir):
    # --- Storing the classifier to disk ---
    rf_dir = "rf_models"
    try:
        if not os.path.exists(os.path.join(price_data_dir, rf_dir)):
            os.makedirs(os.path.join(price_data_dir, rf_dir))

        model_filename = f"{strategy_name}_rf_classifier.pkl"
        model_path = os.path.join(
            price_data_dir, rf_dir, model_filename
        )  # Define where to save

        with open(model_path, "wb") as file:
            pickle.dump(rf_classifier, file)

        logger.info(f"Classifier for strategy {strategy_name} saved to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        return None


def load_rf_model(strategy_name, price_data_dir="PriceData"):
    rf_dir = "rf_models"
    model_filename = f"{strategy_name}_rf_classifier.pkl"
    model_path = os.path.join(price_data_dir, rf_dir, model_filename)

    try:
        with open(model_path, "rb") as file:  # Open in binary read mode ("rb")
            loaded_rf_classifier = pickle.load(file)
        logging.info("Model loaded successfully!")
        return loaded_rf_classifier
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        return None


def get_tables_list(con_source):
    # Get list of all strategies with trades (tables) from source database
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables_list = pd.read_sql(tables_query, con_source)["name"].tolist()
    logger.info(f"{len(tables_list) = }")
    return tables_list


def check_model_exists(strategy_name):
    price_data_dir = "PriceData"
    rf_dir = "rf_models"
    model_filename = f"{strategy_name}_rf_classifier.pkl"
    if not os.path.exists(os.path.join(price_data_dir, rf_dir, model_filename)):
        return False
    return True


def main():
    start_time = time.time()

    price_data_dir = "PriceData"
    trades_list_db_name = os.path.join(price_data_dir, "trades_list_vectorised.db")
    con_tl = sqlite3.connect(trades_list_db_name)

    strategies_list = get_tables_list(con_tl)
    # removes 'summary' table from list if it exists
    if "summary" in strategies_list:
        strategies_list.remove("summary")

    # strategies = [strategies_test[0], strategies_test[1]]
    for idx, strategy in enumerate(strategies_list):
        start_time_strategy = time.time()
        strategy_name = strategy
        logger.info(f"{strategy_name} {idx + 1}/{len(strategies_list)}")

        # check if model already saved
        if check_model_exists(strategy_name):
            logger.info(f"Model for {strategy_name} already exists, skipping...")
            continue

        # get trades from db
        existing_data_query = f"SELECT * FROM {strategy_name}"
        trades_data_df = pd.read_sql(
            existing_data_query, con_tl, index_col=["trade_id"]
        )

        # Ensure trades_data_df is a Pandas DataFrame
        assert isinstance(
            trades_data_df, pd.DataFrame
        ), "trades_data_df is not a Pandas DataFrame"

        # logger.info(f"{len(trades_data_df) = }")
        logger.info(f"{trades_data_df.shape = }")
        if trades_data_df.empty:
            logger.info(f"no trades for {strategy_name}, skipping...")
            continue
        elif len(trades_data_df) < 100:
            logger.info(f"too few trades for {strategy_name}, skipping...")
            continue

        """
        train and save models
        """
        train_model = True
        if train_model:
            try:
                logger.info(f"Training classifier for strategy {strategy_name}")
                rf_classifier = None
                rcv = True
                if rcv:
                    # param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}
                    param_dist = {
                        "n_estimators": randint(50, 500),
                        "max_depth": randint(1, 20),
                    }
                    rf_classifier, accuracy, precision, recall = (
                        train_random_forest_classifier_RandomizedSearchCV(
                            trades_data_df, param_dist
                        )
                    )
                else:
                    rf_classifier, accuracy, precision, recall = (
                        train_random_forest_classifier(trades_data_df)
                    )

                rf_dict = {
                    "rf_classifier": rf_classifier,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }
                logger.info(f"{rf_dict = }")
                assert rf_classifier is not None, "rf_classifier is None"
                logger.info(
                    f"Classifier for strategy {strategy_name} trained successfully."
                )
            except Exception as e:
                logger.error(
                    f"Error training classifier for strategy {strategy_name}: {e}"
                )

            model_path = store_rf_model_to_disk(rf_dict, strategy_name, price_data_dir)
            assert model_path is not None, "Model path is None, model saving failed."

            logger.info(
                f"{strategy_name}: accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}"
            )

        """"
        Load and Predict
        """
        load_model = False
        if load_model:
            loaded_rf_dict = load_rf_model(strategy_name)
            assert (
                loaded_rf_dict is not None
            ), "loaded_model is None, model loading failed."

            if loaded_rf_dict:
                # 2. use rf models to generate predictions df
                # Example usage: Make a prediction for a specific strategy
                sample_data = {"^VIX": [15], "One_day_spy_return": [1.5]}
                sample_df = pd.DataFrame(sample_data, index=[0])

                prediction = predict_random_forest_classifier(
                    loaded_rf_dict["rf_classifier"], sample_df
                )

                accuracy = loaded_rf_dict["accuracy"]
                precision = loaded_rf_dict["precision"]
                recall = loaded_rf_dict["recall"]

                logger.info(
                    f"\nPrediction for {strategy_name}: {prediction}, accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}"
                )

        end_time_strategy = time.time()  # Record the end time
        elapsed_time_strategy = (
            end_time_strategy - start_time_strategy
        )  # Calculate elapsed time
        logger.info(
            f"Execution time for strategy {strategy_name}: {elapsed_time_strategy:.2f} seconds"
        )

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    logger = setup_logging("logs", "rf_models.log", level=logging.info)
    main()
