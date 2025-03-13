import logging
import os
import sys

import certifi
from push import push

# from push import push
from pymongo import MongoClient
from testing import test
from training import train
from variables import config_dict

import wandb

# Local module imports after standard/third-party imports
from config import FINANCIAL_PREP_API_KEY, mongo_url
from control import mode, test_period_end, train_period_start, train_tickers, regime_tickers
from helper_files.client_helper import get_ndaq_tickers, strategies
from TradeSim.utils import initialize_simulation, precompute_strategy_decisions

# Ensure sys.path manipulation is at the top, before other local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ca = certifi.where()

# Set up logging
logs_dir = "logs"
# Create the directory if it doesn't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = logging.FileHandler(os.path.join(logs_dir, "train_test.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if __name__ == "__main__":
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)

    # Initialize W&B run
    wandb.init(
        project=config_dict["project_name"],
        config=config_dict,
        name=config_dict["experiment_name"],
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

    # prepare regime ma calcs eg 1-day spy return. Use pandas dataframe.
    # Maybe create a .get(eg vix) function to complete the trade data list.

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

    if mode == "train":
        train(
            ticker_price_history,
            ideal_period,
            mongo_client,
            precomputed_decisions,
            logger,
        )

    elif mode == "test":
        test(
            ticker_price_history,
            ideal_period,
            mongo_client,
            precomputed_decisions,
            logger,
        )
    elif mode == "push":
        push()
    # elif mode == "push":
    #     push()
