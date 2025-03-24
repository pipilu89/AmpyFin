import logging
import os
import sys
import sqlite3
import pandas as pd
import certifi
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FINANCIAL_PREP_API_KEY, mongo_url
from helper_files.client_helper import get_ndaq_tickers, setup_logging, strategies_test
from TradeSim.utils import precompute_strategy_decisions, load_json_to_dict
from PriceData.store_price_data import sql_to_df_with_date_range
from store_price_data import (
    create_table_schema_strategy_decisions,
    convert_df_to_sql_values,
    sql_to_df_with_date_range_no_index,
    upsert_strategy_decisons,
)

from control import (
    test_period_end,
    train_period_start,
    train_tickers,
    regime_tickers,
)

ca = certifi.where()

# Ensure the PriceData directory exists
price_data_dir = "PriceData"
os.makedirs(price_data_dir, exist_ok=True)

# Database connection
strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
con_sd = sqlite3.connect(strategy_decisions_db_name)

if __name__ == "__main__":
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    list = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
    print(list)
