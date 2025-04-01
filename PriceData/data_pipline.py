"""
1. price data from yf.
db: price_data.db
db format: table: ticker, index/row: date, col: OLCHV
runtime: 5mins

2. strategy decisions.
db: strategy_decisions_intermediate.db => table: ticker, index/row: date, col: OLCHV, strategies
db: strategy_decisions_final.db => db format: table: strategy, index/row: date, col: tickers
runtime: 10mins

3. trades list.
db: trades_list_vectorised.db => table: strategy, index/row: trade_id, col: ticker, buy/sell date, buy/sell price, profit ratio, regime data
runtime: 3-4 hours

4. train random forest models for each strategy, based on trades list.
stored in rf_models folder.
runtime: 2-3 minutes per strategy. 3-5 hours.

5. test model.
create testing_random_forest_v2.py

"""
