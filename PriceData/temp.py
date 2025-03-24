import logging
import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging, strategies_test
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

# Ensure the PriceData directory exists
price_data_dir = "PriceData"
os.makedirs(price_data_dir, exist_ok=True)

# Database connection
strategy_decisions_db_name = os.path.join(price_data_dir, "strategy_decisions.db")
con_sd = sqlite3.connect(strategy_decisions_db_name)

if __name__ == "__main__":
    logger = setup_logging("logs", "store_data.log", level=logging.INFO)
    tickers_list = train_tickers + regime_tickers
    strategies = strategies_test
    # 1. load strategy_decisions from db
    precomputed_decisions = {}
    for strategy in strategies:
        strategy_name = strategy.__name__

        existing_data_query = f"SELECT * FROM {strategy_name}"
        df_existing = pd.read_sql(existing_data_query, con_sd, index_col="Date")

        logger.info(f"{df_existing = }")

        # Melt df_existing to stack ticker columns to rows
        precomputed_decisions = df_existing.reset_index().melt(
            id_vars=["Date"], var_name="Ticker", value_name="Action"
        )

        logger.info(f"{precomputed_decisions = }")
