from polygon import RESTClient
from config import FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, BASE_URL, mongo_url, POLYGON_API_KEY
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from zoneinfo import ZoneInfo
from pymongo import MongoClient
import time
from datetime import datetime, timedelta
import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import (
    StockBarsRequest,
    StockTradesRequest,
    StockQuotesRequest
)
from alpaca.trading.requests import (
    GetAssetsRequest, 
    MarketOrderRequest, 
    LimitOrderRequest, 
    StopOrderRequest, 
    StopLimitOrderRequest, 
    TakeProfitRequest, 
    StopLossRequest, 
    TrailingStopOrderRequest, 
    GetOrdersRequest, 
    ClosePositionRequest
)
from alpaca.trading.enums import ( 
    AssetStatus, 
    AssetExchange, 
    OrderSide, 
    OrderType, 
    TimeInForce, 
    OrderClass, 
    QueryOrderStatus
)
from alpaca.common.exceptions import APIError
from strategies.talib_indicators import *
import math
import yfinance as yf
import logging
from collections import Counter
from trading_client import market_status
from helper_files.client_helper import strategies, get_latest_price, get_ndaq_tickers, dynamic_period_selector
import time
from datetime import datetime 
import heapq 
import certifi
ca = certifi.where()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('rank_system.log'),  # Log messages to a file
        logging.StreamHandler()             # Log messages to the console
    ]
)
from control import time_delta_mode, time_delta_increment, time_delta_multiplicative,time_delta_balanced, rank_liquidity_limit, rank_asset_limit, profit_price_change_ratio_d1, profit_profit_time_d1, profit_price_change_ratio_d2, profit_profit_time_d2, profit_profit_time_else, loss_price_change_ratio_d1, loss_price_change_ratio_d2, loss_profit_time_d1, loss_profit_time_d2, loss_profit_time_else

import json
import pandas as pd

from helper_files.client_helper import market_status as market_status_helper
import traceback
import os
import glob

def process_ticker(ticker, mongo_client, df_historical_single_ticker, current_date, latest_price):
   try:
      
      current_price = None
      current_price = latest_price
      while current_price is None:
         try:
            current_price = get_latest_price(ticker)
         except Exception as fetch_error:
            logging.warning(f"Error fetching price for {ticker}. Retrying... {fetch_error}")
            
            return
      
      indicator_tb = mongo_client.IndicatorsDatabase
      indicator_collection = indicator_tb.Indicators

      actions_dict = {ticker:{'buy': 0, 'sell': 0, 'hold': 0}}
      for strategy in strategies:
         historical_data = None

         while historical_data is None:
            try:
               period = indicator_collection.find_one({'indicator': strategy.__name__})
               if not df_historical_single_ticker.empty:
                  historical_data = df_historical_single_ticker
                  historical_data = adjust_df_length_based_on_period(df_historical_single_ticker, period['ideal_period'], current_date)
                  logging.debug(f"historical_data: {ticker}, {strategy.__name__}, {period['ideal_period']}, {len(historical_data) = }")
               else:
                  historical_data = get_data(ticker, mongo_client, period['ideal_period'])
            except Exception as fetch_error:
               logging.warning(f"Error fetching historical data for {ticker}. Retrying... {fetch_error}")
               time.sleep(60)

         db = mongo_client.trading_simulator  
         holdings_collection = db.algorithm_holdings
         # print(f"Processing {strategy.__name__} for {ticker}")
         logging.debug(f"Processing {strategy.__name__} for {ticker}")
         strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
         if not strategy_doc:
            logging.warning(f"Strategy {strategy.__name__} not found in database. Skipping.")
            continue

         account_cash = strategy_doc["amount_cash"]
         total_portfolio_value = strategy_doc["portfolio_value"]
         # logging.info(f"debug {account_cash = }, {total_portfolio_value = }")
         
         portfolio_qty = strategy_doc["holdings"].get(ticker, {}).get("quantity", 0)

         action, quantity = simulate_trade(ticker, strategy, historical_data, current_price,
                        account_cash, portfolio_qty, total_portfolio_value, mongo_client)
         
         actions_dict[ticker][action] += 1
         
      actions_dict[ticker]["total"] = sum(actions_dict[ticker].values())
      actions_dict[ticker]["current_price"] = current_price
      logging.info(f"{ticker} processing completed. {actions_dict}")
   except Exception as e:
      logging.error(f"Error in thread for {ticker}, {action}, {quantity = }, {current_price = }: {e}")
      logging.error(f"{traceback.format_exc()}, {strategy.__name__}. {ticker}")

def simulate_trade(ticker, strategy, historical_data, current_price, account_cash, portfolio_qty, total_portfolio_value, mongo_client):
   """
   Simulates a trade based on the given strategy and updates MongoDB.
   """
    
   # Simulate trading action from strategy
   # print(f"Simulating trade for {ticker} with strategy {strategy.__name__} and quantity of {portfolio_qty}")
   logging.debug(f"Simulating trade for {ticker} with strategy {strategy.__name__} and quantity of {portfolio_qty}")
   action, quantity = simulate_strategy(strategy, ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value)
   
   # MongoDB setup
   
   db = mongo_client.trading_simulator
   holdings_collection = db.algorithm_holdings
   points_collection = db.points_tally
   
   # Find the strategy document in MongoDB
   strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
   holdings_doc = strategy_doc.get("holdings", {})
   time_delta = db.time_delta.find_one({})['time_delta']
   
   # if action == "buy":
   #    cash_remaining_if_buy = strategy_doc["amount_cash"] - quantity * current_price
   #    pct_of_portfolio_if_buy = ((portfolio_qty + quantity) * current_price) / total_portfolio_value
   #    logging.info(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price} | Strategy: {strategy.__name__}, {cash_remaining_if_buy = }, {pct_of_portfolio_if_buy = }")
   
   # Update holdings and cash based on trade action
   if action in ["buy"] and strategy_doc["amount_cash"] - quantity * current_price > rank_liquidity_limit and quantity > 0 and ((portfolio_qty + quantity) * current_price) / total_portfolio_value < rank_asset_limit:
      logging.debug(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price} | Strategy: {strategy.__name__}")
      # Calculate average price if already holding some shares of the ticker
      if ticker in holdings_doc:
         current_qty = holdings_doc[ticker]["quantity"]
         new_qty = current_qty + quantity
         average_price = (holdings_doc[ticker]["price"] * current_qty + current_price * quantity) / new_qty
      else:
         new_qty = quantity
         average_price = current_price

      # Update the holdings document for the ticker. 
      holdings_doc[ticker] = {
            "quantity": new_qty,
            "price": average_price
      }

      # Deduct the cash used for buying and increment total trades
      holdings_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set": {
                  "holdings": holdings_doc,
                  "amount_cash": strategy_doc["amount_cash"] - quantity * current_price,
                  "last_updated": datetime.now()
            },
            "$inc": {"total_trades": 1}
         },
         upsert=True
      )
      

   elif action in ["sell"] and str(ticker) in holdings_doc and holdings_doc[str(ticker)]["quantity"] > 0:
      
      logging.debug(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price} | Strategy: {strategy.__name__}")
      current_qty = holdings_doc[ticker]["quantity"]
        
      # Ensure we do not sell more than we have
      sell_qty = min(quantity, current_qty)
      holdings_doc[ticker]["quantity"] = current_qty - sell_qty
      
      price_change_ratio = current_price / holdings_doc[ticker]["price"] if ticker in holdings_doc else 1
      
      

      if current_price > holdings_doc[ticker]["price"]:
         #increment successful trades
         holdings_collection.update_one(
            {"strategy": strategy.__name__},
            {"$inc": {"successful_trades": 1}},
            upsert=True
         )
         
         # Calculate points to add if the current price is higher than the purchase price
         if price_change_ratio < profit_price_change_ratio_d1:
            points = time_delta * profit_profit_time_d1
         elif price_change_ratio < profit_price_change_ratio_d2:
            points = time_delta * profit_profit_time_d2
         else:
            points = time_delta * profit_profit_time_else
         
      else:
         # Calculate points to deduct if the current price is lower than the purchase price
         if holdings_doc[ticker]["price"] == current_price:
            holdings_collection.update_one(
               {"strategy": strategy.__name__},
               {"$inc": {"neutral_trades": 1}}
            )
            
         else:   
            
            holdings_collection.update_one(
               {"strategy": strategy.__name__},
               {"$inc": {"failed_trades": 1}},
               upsert=True
            )
         
         if price_change_ratio > loss_price_change_ratio_d1:
            points = -time_delta * loss_profit_time_d1
         elif price_change_ratio > loss_price_change_ratio_d2:
            points = -time_delta * loss_profit_time_d2
         else:
            points = -time_delta * loss_profit_time_else
         
      # Update the points tally
      points_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set" : {
               "last_updated": datetime.now()
            },
            "$inc": {"total_points": points}
         },
         upsert=True
      )
      if holdings_doc[ticker]["quantity"] == 0:      
         del holdings_doc[ticker]
      # Update cash after selling
      holdings_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set": {
               "holdings": holdings_doc,
               "amount_cash": strategy_doc["amount_cash"] + sell_qty * current_price,
               "last_updated": datetime.now()
            },
            "$inc": {"total_trades": 1}
         },
         upsert=True
      )

        
      # Remove the ticker if quantity reaches zero
      if holdings_doc[ticker]["quantity"] == 0:      
         del holdings_doc[ticker]
        
   else:
      logging.debug(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price} | Strategy: {strategy.__name__}")
   # print(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}")
   # Close the MongoDB connection
   return action, quantity

def update_portfolio_values(client):
   """
   still need to implement.
   we go through each strategy and update portfolio value buy cash + summation(holding * current price)
   """
   
   db = client.trading_simulator  
   holdings_collection = db.algorithm_holdings
   # Update portfolio values
   for strategy_doc in holdings_collection.find({}):
      # Calculate the portfolio value for the strategy
      portfolio_value = strategy_doc["amount_cash"]
      
      for ticker, holding in strategy_doc["holdings"].items():
         # The current price can be gotten through a cache system maybe
         # if polygon api is getting clogged - but that hasn't happened yet
         # Also implement in C++ or C instead of python
         # Get the current price of the ticker from the Polygon API
         # Use a cache system to store the latest prices
         # If the cache is empty, fetch the latest price from the Polygon API
         # Cache should be updated every 60 seconds 
         current_price = None
         while current_price is None:
            try:
               # get latest price shouldn't cache - we should also do a delay
               current_price = get_latest_price(ticker)
            except:
               print(f"Error fetching price for {ticker}. Retrying...")
               break
               
               # Will sleep 120 seconds before retrying to get latest price
         # print(f"Current price of {ticker}: {current_price}")
         logging.debug(f"Current price of {ticker}: {current_price}")
         if current_price is None:
            current_price = 0
         # Calculate the value of the holding
         holding_value = holding["quantity"] * current_price
         if current_price == 0:
            holding_value = 5000
         # Add the holding value to the portfolio value
         portfolio_value += holding_value
          
      # Update the portfolio value in the strategy document
      holdings_collection.update_one({"strategy": strategy_doc["strategy"]}, {"$set": {"portfolio_value": portfolio_value}}, upsert=True)

   # Update MongoDB with the modified strategy documents
   

def update_ranks(client):
   """"
   based on portfolio values, rank the strategies to use for actual trading_simulator
   """
   
   db = client.trading_simulator
   points_collection = db.points_tally
   rank_collection = db.rank
   algo_holdings = db.algorithm_holdings
   """
   delete all documents in rank collection first
   """
   rank_collection.delete_many({})
   """
   Reason why delete rank is so that rank is intially null and
   then we can populate it in the order we wish
   now update rank based on successful_trades - failed
   """
   q = []
   for strategy_doc in algo_holdings.find({}):
      """
      based on (points_tally (less points pops first), failed-successful(more negtive pops first), portfolio value (less value pops first), and then strategy_name), we add to heapq.
      """
      strategy_name = strategy_doc["strategy"]
      if strategy_name == "test" or strategy_name == "test_strategy":
         continue
      if points_collection.find_one({"strategy": strategy_name})["total_points"] > 0:
         
         heapq.heappush(q, (points_collection.find_one({"strategy": strategy_name})["total_points"] * 2 + (strategy_doc["portfolio_value"]), strategy_doc["successful_trades"] - strategy_doc["failed_trades"], strategy_doc["amount_cash"], strategy_doc["strategy"]))
      else:
         heapq.heappush(q, (strategy_doc["portfolio_value"], strategy_doc["successful_trades"] - strategy_doc["failed_trades"], strategy_doc["amount_cash"], strategy_doc["strategy"]))
   rank = 1
   while q:
      
      _, _, _, strategy_name = heapq.heappop(q)
      rank_collection.insert_one({"strategy": strategy_name, "rank": rank})
      rank+=1
   
   """
   Delete historical database so new one can be used tomorrow
   """
   db = client.HistoricalDatabase
   collection = db.HistoricalDatabase
   collection.delete_many({})
   logging.info("Successfully updated ranks")
   logging.info("Successfully deleted historical database")
   
def get_historical_data_from_yf(ndaq_tickers, period):
    logging.info('download historical price data from yf...')
    try:
        df = yf.download(ndaq_tickers, group_by='Ticker', period=period, interval='1d', auto_adjust=True, repair=True, rounding=True)
        return df
    except Exception as e:
        logging.error(f'yf error: {e}')
        return None

def adjust_df_length_based_on_period(df, period, current_date):
   adjustment = 1.1 #increase the length of the dataframe to avoid missing data from holidays etc.
   if period == '1mo':
      x = 30
   elif period == '3mo':
      x = 90
   elif period == '6mo':
      x = 180
   elif period == '1y':
      x = 365
   # elif period == '2y':
   #    start_date = current_date - timedelta(days=730)
   if period == '2y':
      return df
   else:
      df = df.tail(x)
   return df
        
def dl_historical_data_and_batch_insert_into_mdb(mongo_client, ndaq_tickers, period_list, current_date):
   db = mongo_client.HistoricalDatabase
   collection = db.HistoricalDatabase

   logging.info("Deleting historical database...")
   try:
      collection.delete_many({})
      logging.info("Successfully deleted historical database")
   except Exception as e:
      logging.error(f"Error deleting historical database: {e}")
      raise

   period_max ='2y'

   # use yf dowloaded data, split period_max('2y') into shorter periods, rather than dl multiple periods. Possible problem if ticker does have 2 years of data (it may have 1mo or 6mo). Test with recent ipo.
   df = get_historical_data_from_yf(ndaq_tickers, period_max)
   if not df.empty:
      # Stack multi-level column index
      df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
      df = df[['Ticker','Open', 'High', 'Low', 'Close', 'Volume']]

      documents = []
      for period in period_list:
         for ticker in ndaq_tickers:
            df_single_ticker = df.loc[df['Ticker'] == ticker]
            df_single_ticker = df_single_ticker.dropna()

            df_single_ticker = adjust_df_length_based_on_period(df_single_ticker, period, current_date)

            records = df_single_ticker.reset_index().to_dict('records')
            documents.append({"ticker": ticker, "period": period, 'data': records})
      if documents:
         collection.insert_many(documents)
         # logging.info(f"Data for period {period} stored in MongoDB. {len(df_single_ticker) = }.")
         logging.info(f"Data for period {period_list} stored in MongoDB.")
         logging.debug(f"{df_single_ticker.head(1) = }")
         logging.debug(f"{df_single_ticker.tail(1) = }")
   return df
 
def get_latest_prices_from_yf(tickers):
    """
    Batch download the latest prices for multiple tickers using yfinance.
    
    Parameters:
    - tickers (list): List of ticker symbols.
    
    Returns:
    - dict: A dictionary with ticker symbols as keys and their latest prices as values.
    """
    logging.info(f"Batch downloading latest prices from yfinance... {len(tickers) = }")
    try:
        # Download the latest prices for the tickers
        df = yf.download(tickers, period='1d', interval='1m', group_by='Ticker', auto_adjust=True, repair=True, rounding=True)
        
        # Extract the latest prices
        latest_prices = {}
        for ticker in tickers:
            if ticker in df.columns.levels[0]:
                latest_prices[ticker] = df[ticker]['Close'].iloc[-1]
            else:
                latest_prices[ticker] = None
        logging.info(f"{len(latest_prices) = }")
        logging.info(f"{latest_prices}")
        return latest_prices
    except Exception as e:
        logging.error(f"Error downloading latest prices from yfinance: {e}")
        return {}

def get_latest_prices_from_yf2(tickers):
      """
      Batch download the latest prices for multiple tickers using yfinance.
      
      Parameters:
      - tickers (list): List of ticker symbols.
      
      Returns:
      - dict: A dictionary with ticker symbols as keys and their latest prices as values.
      """
      logging.info(f"Batch downloading latest prices from yfinance... {len(tickers) = }")
      try:
         # Download the latest prices for the tickers
         df = yf.download(tickers, period='1d', interval='1m', group_by='Ticker', auto_adjust=True, repair=True, rounding=True)
         if not df.empty:
            # truncate df to most recent date only
            df = df[df.index == df.index.max()]
            # Stack multi-level column index
            df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            df = df[['Ticker','Open', 'High', 'Low', 'Close', 'Volume']]
            return df
      except Exception as e:
         logging.error(f"Error downloading latest prices from yfinance: {e}")
         return {}
      
def clean_old_files(directory, pattern, max_files):
   files = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
   while len(files) > max_files:
      file_to_delete = files.pop(0)
      os.remove(file_to_delete)
      logging.info(f"Deleted old file: {file_to_delete}")

def main():  
   """  
   Main function to control the workflow based on the market's status.  
   """  
   
   ndaq_tickers = []  
   early_hour_first_iteration = True
   post_market_hour_first_iteration = True
   status_previous = None
   count = 0
   period_list = ["1mo", "3mo", "6mo", "1y", "2y"]
   df_historical_yf_prices = pd.DataFrame()
   df_latest_prices_previous = pd.DataFrame()

   today_date_str = datetime.now().strftime('%Y-%m-%d')
   historical_data_filename = f'df_historical_yf_prices_{today_date_str}.csv'
   historical_data_directory = '.'  # Directory where the historical data files are stored

   while True: 
      mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
   
      # Get the market status from the Polygon API
      # client = RESTClient(api_key=POLYGON_API_KEY)
      # status = market_status(client)  # Use the helper function for market status
      status = "open"

      # status = mongo_client.market_data.market_status.find_one({})["market_status"] # orig update status
      if status != status_previous:
         logging.info(f"Market status: {status}")
      status_previous = status
   
      if status == "open":  
         # Connection pool is not thread safe. Create a new client for each thread.
         # We can use ThreadPoolExecutor to manage threads - maybe use this but this risks clogging
         # resources if we have too many threads or if a thread is on stall mode
         # We can also use multiprocessing.Pool to manage threads
         current_date = datetime.now()
         logging.info("Market is open. Processing strategies.")  
      
         if not ndaq_tickers:
            ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
            # ndaq_tickers = ["AAPL", "AMD", "PPP"]

         # batch download ticker data from yfinance prior to threading
         if df_historical_yf_prices.empty:            
            if not os.path.exists(historical_data_filename):
               df_historical_yf_prices = dl_historical_data_and_batch_insert_into_mdb(mongo_client, ndaq_tickers, period_list, current_date)
               # store df_historical_yf_prices in local file
               df_historical_yf_prices.to_csv(historical_data_filename, index=True)
               logging.info(f"Downloaded and saved df_historical_yf_prices to {historical_data_filename}")
               # Clean old files to maintain a maximum of 5 files
               clean_old_files(historical_data_directory, 'df_historical_yf_prices_*.csv', 5)
            else:
               # load df_historical_yf_prices from local file
               try:
                  df_historical_yf_prices = pd.read_csv(historical_data_filename)
                  logging.info(f"Loaded df_historical_yf_prices from {historical_data_filename}. {len(df_historical_yf_prices) = }")
               except Exception as e:
                  logging.error(f"Error loading {historical_data_filename}: {e}")

         # batch download current prices from yf. This is to avoid multiple calls to yf in the threads.
         df_latest_prices = get_latest_prices_from_yf2(ndaq_tickers)
         # save df to local file
         df_latest_prices.to_csv('latest_prices_yf.csv', index=True)
       
         threads = []

         for ticker in ndaq_tickers:
            df_single_ticker = df_historical_yf_prices.loc[df_historical_yf_prices['Ticker'] == ticker]
            df_single_ticker = df_single_ticker.dropna()
            latest_price = df_latest_prices.loc[df_latest_prices['Ticker'] == ticker, 'Close'].values[0]
            # logging.info(f"{latest_price = }")
            # check if latest price is None or NaN
            if latest_price is None or math.isnan(latest_price):
               logging.warning(f"Latest price for {ticker}: {latest_price} is invalid (None or NaN). Skipping...")
               continue
            # check if price has changed. if true process ticker. if false, skip.
            if not df_latest_prices_previous.empty:
               if latest_price == df_latest_prices_previous.loc[df_latest_prices_previous['Ticker'] == ticker, 'Close'].values[0]:
                  logging.info(f"Price for {ticker} has not changed. Skipping...")
                  continue

            thread = threading.Thread(target=process_ticker, args=(ticker, mongo_client, df_single_ticker, current_date, latest_price))
            threads.append(thread)
            thread.start()

         # Wait for all threads to complete
         for thread in threads:
            thread.join()

      
         logging.info(f"Finished processing all strategies. Waiting for 30 seconds. {count = }")
         df_latest_prices_previous = df_latest_prices
         count += 1
         time.sleep(30)  
   
      elif status == "early_hours":  
            # During early hour, currently we only support prep
            # However, we should add more features here like premarket analysis
         
            if early_hour_first_iteration is True:  
            
               ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)  
               early_hour_first_iteration = False  
               post_market_hour_first_iteration = True
               logging.info("Market is in early hours. Waiting for 30 seconds.")  
            time.sleep(30)  

      elif status == "closed":  
         # Performs post-market analysis for next trading day
         # Will only run once per day to reduce clogging logging
         # Should self-implementing a delete log process after a certain time - say 1 year
      
         if post_market_hour_first_iteration is True:
            early_hour_first_iteration = True
            logging.info("Market is closed. Performing post-market analysis.") 
            post_market_hour_first_iteration = False
            # Update time delta based on the mode
            
            if time_delta_mode == 'additive':
               mongo_client.trading_simulator.time_delta.update_one({}, {"$inc": {"time_delta": time_delta_increment}})
            elif time_delta_mode == 'multiplicative':
               mongo_client.trading_simulator.time_delta.update_one({}, {"$mul": {"time_delta": time_delta_multiplicative}})
            elif time_delta_mode == 'balanced':
               """
               retrieve time_delta first
               """
               time_delta = mongo_client.trading_simulator.time_delta.find_one({})['time_delta']
               mongo_client.trading_simulator.time_delta.update_one({}, {"$inc": {"time_delta": time_delta_balanced * time_delta}})
         
            #Update ranks
            update_portfolio_values(mongo_client)
            # We keep reusing the same mongo client and never close to reduce the number within the connection pool

            update_ranks(mongo_client)
         time.sleep(60)  
      else:  
         logging.error("An error occurred while checking market status.")  
         time.sleep(60)
      mongo_client.close()
   
   
  
if __name__ == "__main__": 
   
   main()