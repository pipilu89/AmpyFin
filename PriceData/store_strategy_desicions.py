import logging, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_files.client_helper import setup_logging
from TradeSim.utils import precompute_strategy_decisions, load_json_to_dict
from PriceData.store_price_data import sql_to_df_with_date_range

from control import (
    test_period_end,
    train_period_start,
    train_tickers,
    regime_tickers,
)

logger = setup_logging("logs", "store_data.log", level=logging.INFO)

# create ticker price history from db.
# table_name = "AAPL"
# begin_date = "2021-01-04"
# end_date = "2021-01-29"
tickers_list = train_tickers + regime_tickers

ticker_price_history = {}
for ticker in tickers_list:
    ticker_price_history[ticker] = sql_to_df_with_date_range(
        ticker, train_period_start, test_period_end
    )


# load/save local copy of ideal_period
ideal_period_dir = "results"
ideal_period_filename = f"ideal_period.json"
ideal_period, _ = load_json_to_dict(ideal_period_dir, ideal_period_filename)
logger.info(f"ideal_period: {ideal_period}")

strategies = []

# precomputed_decisions = precompute_strategy_decisions(
# strategies,
# ticker_price_history,
# train_tickers,
# ideal_period,
# train_period_start,
# test_period_end,
# logger,
# )
