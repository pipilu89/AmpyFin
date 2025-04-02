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
        logger.info("Model loaded successfully!")
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


def train_rf_model(trades_data_df, strategy_name):
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
            rf_classifier, accuracy, precision, recall = train_random_forest_classifier(
                trades_data_df
            )

        rf_dict = {
            "rf_classifier": rf_classifier,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
        logger.info(f"{rf_dict = }")
        assert rf_classifier is not None, "rf_classifier is None"
        logger.info(f"Classifier for strategy {strategy_name} trained successfully.")
        logger.info(
            f"{strategy_name}: accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}"
        )
        return rf_dict
    except Exception as e:
        logger.error(f"Error training classifier for strategy {strategy_name}: {e}")


def main():
    start_time = time.time()

    price_data_dir = "PriceData"
    trades_list_db_name = os.path.join(price_data_dir, "trades_list_vectorised.db")
    con_tl = sqlite3.connect(trades_list_db_name)

    """
    Choose options.
    eg: load and predict or train and save model.
    """
    train_model = False
    save_model = False
    overwrite_model = False  # if model file exist, do we retrain and save model?
    load_model = True
    model_predict = True  # need to either train or load a model first

    # data for prediction
    sample_data = {"^VIX": [15], "One_day_spy_return": [1.5]}
    sample_df = pd.DataFrame(sample_data, index=[0])

    prediction_results = []

    strategies_list = get_tables_list(con_tl)
    # removes 'summary' table from list if it exists
    if "summary" in strategies_list:
        strategies_list.remove("summary")

    # strategies_list = [strategies_list[-1]]
    # strategies = [strategies_test[0], strategies_test[1]]
    for idx, strategy in enumerate(strategies_list):
        start_time_strategy = time.time()
        strategy_name = strategy
        logger.info(f"{strategy_name} {idx + 1}/{len(strategies_list)}")

        # check if model already saved to help prevent accidental overwrite.
        if check_model_exists(strategy_name):
            if overwrite_model and save_model:
                logger.warning(f"Model file exists, overwriting...")
            elif not overwrite_model and save_model:
                logger.info(f"Model for {strategy_name} already exists, skipping...")
                continue

        """
        train, save, Load, and Predict models.
        """

        if train_model:
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

            rf_dict = train_rf_model(trades_data_df, strategy_name)

        if save_model:
            model_path = store_rf_model_to_disk(rf_dict, strategy_name, price_data_dir)
            assert model_path is not None, "Model path is None, model saving failed."
        if load_model:
            if not check_model_exists(strategy_name):
                logger.info(f"Model for {strategy_name} does not exist, skipping...")
                continue
            rf_dict = load_rf_model(strategy_name)
            if rf_dict is None:
                logger.info(
                    f"Model for {strategy_name} could not be loaded, skipping..."
                )
                continue
            assert isinstance(
                rf_dict, dict
            ), "loaded_model is not a dictionary, model loading failed."

        if model_predict and rf_dict is not None:
            #  use rf models to generate predictions df
            prediction = predict_random_forest_classifier(
                rf_dict["rf_classifier"], sample_df
            )

            accuracy = round(rf_dict["accuracy"], 2)
            precision = round(rf_dict["precision"], 2)
            recall = round(rf_dict["recall"], 2)
            logger.info(
                f"\nPrediction for {strategy_name}: {prediction}, accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}"
            )

            prediction_results.append(
                [strategy_name, prediction, accuracy, precision, recall]
            )

        end_time_strategy = time.time()  # Record the end time
        elapsed_time_strategy = (
            end_time_strategy - start_time_strategy
        )  # Calculate elapsed time
        logger.info(
            f"Execution time for strategy {strategy_name}: {elapsed_time_strategy:.2f} seconds"
        )

    if prediction_results:
        prediction_results_df = pd.DataFrame(
            prediction_results,
            columns=["strategy", "prediction", "accuracy", "precision", "recall"],
        )
        logger.info(f"{prediction_results_df}")
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    logger = setup_logging("logs", "rf_models.log", level=logging.INFO)
    main()
