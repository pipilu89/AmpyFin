from datetime import datetime
import logging
import os
import sys

import certifi
import pandas as pd
from push import push

# from push import push
from pymongo import MongoClient
from testing import test
from training import train
from variables import config_dict

import wandb

# Ensure sys.path manipulation is at the top, before other local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local module imports after standard/third-party imports
from config import FINANCIAL_PREP_API_KEY, mongo_url, environment
from control import (
    mode,
    test_period_end,
    train_period_start,
    train_tickers,
    regime_tickers,
)
from helper_files.client_helper import (
    get_ndaq_tickers,
    load_json_to_dict,
    store_dict_as_json,
    save_df_to_csv,
    strategies,
)
from TradeSim.utils import (
    initialize_simulation,
    precompute_strategy_decisions,
    prepare_regime_data,
)
from TradeSim.testing_random_forest import test_random_forest

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

if __name__ == "__main__":
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    today_date_str = datetime.now().strftime("%Y-%m-%d")

    # Initialize W&B run
    if environment == "dev":
        wandb_mode = "disabled"
        # wandb_mode = 'offline'
        # wandb_mode = 'dryrun'
        # wandb_mode = 'async'
    else:
        wandb_mode = "online"

    wandb.init(
        project=config_dict["project_name"],
        config=config_dict,
        name=config_dict["experiment_name"],
        mode=wandb_mode,
    )

    # If no tickers provided, fetch Nasdaq tickers
    if not train_tickers:
        logger.info("No tickers provided. Fetching Nasdaq tickers...")
        train_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
        logger.info(f"Fetched {len(train_tickers)} tickers.")

    # new combine train and regime tickers
    train_regime_tickers = train_tickers + regime_tickers

    # Initialize simulation
    ticker_price_history, ideal_period = initialize_simulation(
        train_period_start,
        test_period_end,
        train_regime_tickers,
        mongo_client,
        FINANCIAL_PREP_API_KEY,
        logger,
    )

    # === prepare REGIME ma calcs eg 1-day spy return. Use pandas dataframe.
    ticker_price_history = prepare_regime_data(ticker_price_history, logger)

    precomputed_decisions_filename = f"precomputed_decisions_{today_date_str}.csv"
    precomputed_decisions_filepath = os.path.join(
        results_dir, precomputed_decisions_filename
    )
    if not os.path.exists(precomputed_decisions_filepath):

        # Precompute all strategy decisions
        precomputed_decisions = precompute_strategy_decisions(
            strategies,
            ticker_price_history,
            train_tickers,
            ideal_period,
            train_period_start,
            test_period_end,
            logger,
        )
        # store_dict_as_json(
        #     precomputed_decisions, precomputed_decisions_filename, results_dir, logger
        # )
        save_df_to_csv(
            precomputed_decisions, precomputed_decisions_filename, results_dir, logger
        )

    else:
        # load from local file
        precomputed_decisions = pd.read_csv(precomputed_decisions_filepath)
        # precomputed_decisions, _ = load_json_to_dict(
        #     results_dir, precomputed_decisions_filename
        # )

    if mode == "train":
        train(
            ticker_price_history,
            ideal_period,
            mongo_client,
            precomputed_decisions,
            logger,
        )

    elif mode == "test":
        test_random_forest(
            ticker_price_history,
            ideal_period,
            mongo_client,
            precomputed_decisions,
            # strategies,
            logger,
        )
        # test(
        #     ticker_price_history,
        #     ideal_period,
        #     mongo_client,
        #     precomputed_decisions,
        #     logger,
        # )
    elif mode == "push":
        push()
    else:
        logger.error(f"Invalid mode. Exiting. {mode = }")
