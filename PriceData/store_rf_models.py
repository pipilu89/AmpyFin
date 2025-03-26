import logging
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from random_forest import train_and_store_classifiers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test
from TradeSim.utils import (
    prepare_regime_data,
    simulate_trading_day,
)
from PriceData.store_price_data import (
    sql_to_df_with_date_range,
)
from config import PRICE_DB_PATH
from control import (
    test_period_end,
    train_period_start,
    train_period_end,
    train_tickers,
    regime_tickers,
    train_time_delta,
)


def main():
    # 1. use trade lists to train rf models
    trained_classifiers, strategies_with_enough_data = train_and_store_classifiers(
        trades_data_df, logger
    )
    # 2. use rf models to generate predictions df


if __name__ == "__main__":
    logger = setup_logging("logs", "rf_models.log", level=logging.info)
    main()
