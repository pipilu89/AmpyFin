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

    existing_data_query = f"SELECT * FROM 'MIDPRICE_indicator'"
    df1 = pd.read_sql(existing_data_query, con_sd, index_col="Date")
    print(df1)
    # result = df1.to_dict(orient="split")
    # print(result)

    # Example data dictionary
    data_dict = {
        "Date": ["2024-12-23", "2024-12-24", "2024-12-26"],
        "GOOGL": ["Buy", "Buy", "Buy"],
        "MSFT": ["Buy", "Sell", "Hold"],
    }
    df2 = pd.DataFrame.from_dict(data_dict)
    df2.set_index("Date", inplace=True)
    print(df2)

# Merge new data with existing data on index (Date)
df_merged = pd.merge(
    df1,
    df2,
    left_index=True,
    right_index=True,
    how="outer",
    suffixes=("_left", "_right"),
)

# Resolve conflicts for all columns
for column in df2.columns:
    if column in df1.columns:
        df_merged[column] = df_merged[f"{column}_right"].combine_first(
            df_merged[f"{column}_left"]
        )
        df_merged.drop(columns=[f"{column}_left", f"{column}_right"], inplace=True)

print(df_merged)
