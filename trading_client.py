from polygon import RESTClient
from config import POLYGON_API_KEY, FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, BASE_URL, mongo_url, environment
import json
import certifi
from urllib.request import urlopen
from zoneinfo import ZoneInfo
from pymongo import MongoClient
import time
from datetime import datetime, timedelta
from helper_files.client_helper import place_order, get_ndaq_tickers, market_status, strategies, get_latest_price, dynamic_period_selector
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from strategies.archived_strategies.trading_strategies_v1 import get_historical_data
import yfinance as yf
import logging
from collections import Counter
from statistics import median, mode
import statistics
import heapq
import requests
from strategies.talib_indicators import *
import threading
import sys
from bson.decimal128 import Decimal128

from control import trade_liquidity_limit, trade_asset_limit, suggestion_heap_limit, fractional_shares

buy_heap = []
suggestion_heap = []
sold = False

ca = certifi.where()

def weighted_majority_decision_and_median_quantity(decisions_and_quantities):  
    """  
    Determines the majority decision (buy, sell, or hold) and returns the weighted median quantity for the chosen action.  
    Groups 'strong buy' with 'buy' and 'strong sell' with 'sell'.
    Applies weights to quantities based on strategy coefficients.  
    """  
    buy_decisions = ['buy', 'strong buy']  
    sell_decisions = ['sell', 'strong sell']  

    weighted_buy_quantities = []
    weighted_sell_quantities = []
    buy_weight = 0
    sell_weight = 0
    hold_weight = 0
    
    # Process decisions with weights
    for decision, quantity, weight in decisions_and_quantities:
        if decision in buy_decisions:
            weighted_buy_quantities.extend([quantity])
            buy_weight += weight
        elif decision in sell_decisions:
            weighted_sell_quantities.extend([quantity])
            sell_weight += weight
        elif decision == 'hold':
            hold_weight += weight
    
    # Determine the majority decision based on the highest accumulated weight
    if buy_weight > sell_weight and buy_weight > hold_weight:
        return 'buy', median(weighted_buy_quantities) if weighted_buy_quantities else 0, buy_weight, sell_weight, hold_weight
    elif sell_weight > buy_weight and sell_weight > hold_weight:
        return 'sell', median(weighted_sell_quantities) if weighted_sell_quantities else 0, buy_weight, sell_weight, hold_weight
    else:
        return 'hold', 0, buy_weight, sell_weight, hold_weight

def process_ticker(ticker, client, trading_client, data_client, mongo_client, strategy_to_coefficient):
    global buy_heap
    global suggestion_heap
    global sold
    if sold is True:
        logging.info("Sold boolean is True. Exiting process_ticker function.")
    else:
        try:
            decisions_and_quantities = []
            current_price = None
            
            while current_price is None:
                try:
                    current_price = get_latest_price(ticker)
                except Exception as fetch_error:
                    logging.warning(f"Error fetching price for {ticker}. Retrying... {fetch_error}")
                    break
            if current_price is None:
                return
            logging.debug(f"Current price of {ticker}: {current_price}")

            asset_collection = mongo_client.trades.assets_quantities
            limits_collection = mongo_client.trades.assets_limit
            account = trading_client.get_account()
            buying_power = float(account.cash)
            portfolio_value = float(account.portfolio_value)
            cash_to_portfolio_ratio = buying_power / portfolio_value

            asset_info = asset_collection.find_one({'symbol': ticker})
            
            portfolio_qty = asset_info['quantity'] if asset_info else 0.0
           
            # fractional shares
            if fractional_shares == True:
                if isinstance(portfolio_qty, Decimal128):
                    # Convert Decimal128 to decimal.Decimal
                    portfolio_qty = portfolio_qty.to_decimal()
                # Convert decimal.Decimal to float
                portfolio_qty = float(portfolio_qty)

            logging.debug(f"Portfolio quantity for {ticker}: {portfolio_qty}")

            limit_info = limits_collection.find_one({'symbol': ticker})
            if limit_info:
                stop_loss_price = limit_info['stop_loss_price']
                take_profit_price = limit_info['take_profit_price']
                if current_price <= stop_loss_price or current_price >= take_profit_price:
                    sold = True
                    logging.info(f"Executing SELL order for {ticker} due to stop-loss or take-profit condition")
                    quantity = portfolio_qty
                    order = place_order(trading_client, symbol=ticker, side=OrderSide.SELL, quantity=quantity, mongo_client=mongo_client)
                    logging.info(f"Executed SELL order for {ticker}: {order}")
                    return

            indicator_tb = mongo_client.IndicatorsDatabase
            indicator_collection = indicator_tb.Indicators

            for strategy in strategies:
                historical_data = None
                while historical_data is None:
                    try:
                        period = indicator_collection.find_one({'indicator': strategy.__name__})
                        historical_data = get_data(ticker, mongo_client, period['ideal_period'])
                    except Exception as fetch_error:
                        logging.warning(f"Error fetching historical data for {ticker}. Retrying... {fetch_error}")
                        time.sleep(60)

                decision, quantity, _ = simulate_strategy(strategy, ticker, current_price, historical_data, buying_power, portfolio_qty, portfolio_value)
                logging.debug(f"Strategy: {strategy.__name__}, Decision: {decision}, Quantity: {quantity} for {ticker}")
                weight = strategy_to_coefficient[strategy.__name__]
                decisions_and_quantities.append((decision, quantity, weight))

            decision, quantity, buy_weight, sell_weight, hold_weight = weighted_majority_decision_and_median_quantity(decisions_and_quantities)
            logging.info(f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}, Weights: Buy: {buy_weight}, Sell: {sell_weight}, Hold: {hold_weight}")

            if decision == "buy" and float(account.cash) > trade_liquidity_limit and (((quantity + portfolio_qty) * current_price) / portfolio_value) < trade_asset_limit:
                heapq.heappush(buy_heap, (-(buy_weight-(sell_weight + (hold_weight * 0.5))), quantity, ticker))
            elif decision == "sell" and portfolio_qty > 0:
                logging.info(f"Executing SELL order for {ticker} {quantity = }")
                sold = True
                
                # fractional shares
                if fractional_shares == True:
                    ...
                else:
                    quantity = max(quantity, 1) 

                order = place_order(trading_client, symbol=ticker, side=OrderSide.SELL, quantity=quantity, mongo_client=mongo_client)
                logging.info(f"Executed SELL order for {ticker}: {order}")
            elif portfolio_qty == 0.0 and buy_weight > sell_weight and (((quantity + portfolio_qty) * current_price) / portfolio_value) < trade_asset_limit and float(account.cash) > trade_liquidity_limit:
                max_investment = portfolio_value * trade_asset_limit
                
                if fractional_shares == True:
                    ...
                    buy_quantity = min(math.floor((max_investment / current_price)*100)/100, math.floor((buying_power / current_price)*100)/100)
                else:
                    buy_quantity = min(int(max_investment // current_price), int(buying_power // current_price))
                
                if buy_weight > suggestion_heap_limit:

                    if fractional_shares == True:
                        buy_quantity = buy_quantity / 2
                    else:
                        buy_quantity = max(buy_quantity, 2)
                        buy_quantity = buy_quantity // 2
                    
                    logging.info(f"Suggestions for buying for {ticker} with a weight of {buy_weight} and quantity of {buy_quantity}")
                    heapq.heappush(suggestion_heap, (-(buy_weight - sell_weight), buy_quantity, ticker))
                else:
                    logging.info(f"Holding for {ticker}, no action taken.")
            else:
                logging.info(f"Holding for {ticker}, no action taken.")
        
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")

def main():
    """
    Main function to control the workflow based on the market's status.
    """
    
    logging.info("Trading mode is live.")
    global buy_heap
    global suggestion_heap
    global sold
    starting_cash = 1000
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_hour_first_iteration = True
    client = RESTClient(api_key=POLYGON_API_KEY)
    trading_client = TradingClient(API_KEY, API_SECRET)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    db = mongo_client.trades
    asset_collection = db.assets_quantities
    limits_collection = db.assets_limit
    strategy_to_coefficient = {}
    sold = False
    while True:
        client = RESTClient(api_key=POLYGON_API_KEY)
        trading_client = TradingClient(API_KEY, API_SECRET)
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        db = mongo_client.trades
        asset_collection = db.assets_quantities
        limits_collection = db.assets_limit
        market_db = mongo_client.market_data
        market_collection = market_db.market_status
        indicator_tb = mongo_client.IndicatorsDatabase
        indicator_collection = indicator_tb.Indicators
        
        # status = market_status(client)  # Use the helper function for market status
        status = market_status(client) if environment != "dev" else "open"

        # status = "open" if dev else market_status(client)

        # status = "open"
        # market_collection.update_one({}, {"$set": {"market_status": status}})
        
        if status == "open":
            if not ndaq_tickers:
                logging.info("Market is open")
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    
                    rank = rank_collection.find_one({'strategy': strategy.__name__})['rank']
                    coefficient = r_t_c_collection.find_one({'rank': rank})['coefficient']
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
            trading_client = TradingClient(API_KEY, API_SECRET)
            account = trading_client.get_account()
            buying_power = float(account.cash)
            portfolio_value = float(account.portfolio_value)
            cash_to_portfolio_ratio = buying_power / portfolio_value
            qqq_latest = get_latest_price('QQQ')
            spy_latest = get_latest_price('SPY')
            buy_heap = []
            suggestion_heap = []

            trades_db = mongo_client.trades
            portfolio_collection = trades_db.portfolio_values

            portfolio_collection.update_one({"name" : "portfolio_percentage"}, {"$set": {"portfolio_value": (portfolio_value-starting_cash)/starting_cash}})
            portfolio_collection.update_one({"name" : "ndaq_percentage"}, {"$set": {"portfolio_value": (qqq_latest-518.58)/518.58}})
            portfolio_collection.update_one({"name" : "spy_percentage"}, {"$set": {"portfolio_value": (spy_latest-591.95)/591.95}})

            threads = []

            for ticker in ndaq_tickers:
                thread = threading.Thread(target=process_ticker, args=(ticker, client, trading_client, data_client, mongo_client, strategy_to_coefficient))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            trading_client = TradingClient(API_KEY, API_SECRET)
            account = trading_client.get_account()
            while (buy_heap or suggestion_heap) and float(account.cash) > trade_liquidity_limit and sold is False:
                try:
                    trading_client = TradingClient(API_KEY, API_SECRET)
                    account = trading_client.get_account()
                    logging.info(f"Cash: {account.cash}")
                    if buy_heap and float(account.cash) > trade_liquidity_limit:
                        
                        _, quantity, ticker = heapq.heappop(buy_heap)
                        logging.info(f"Executing BUY order (from buy_heap) for {ticker} of quantity {quantity} {buy_heap = }")
                        
                        order = place_order(trading_client, symbol=ticker, side=OrderSide.BUY, quantity=quantity, mongo_client=mongo_client)
                        logging.info(f"Executed BUY order (from buy_heap) for {ticker}: {order}")
                        
                    elif suggestion_heap and float(account.cash) > trade_liquidity_limit:
                        
                        _, quantity, ticker = heapq.heappop(suggestion_heap)
                        logging.info(f"Executing BUY order (from suggestion_heap) for {ticker} of quantity {quantity}\n{len(suggestion_heap) = }\n{suggestion_heap = }")
                        
                        order = place_order(trading_client, symbol=ticker, side=OrderSide.BUY, quantity=quantity, mongo_client=mongo_client)
                        logging.info(f"Executed BUY order (from suggestion_heap) for {ticker}: \n{order}")
                        
                    time.sleep(5)
                    """
                    This is here so order will propage through and we will have an accurate cash balance recorded
                    """
                # except:
                except Exception as e:
                    logging.error(f"Error occurred while executing buy order. Continuing... {e}")
                    break
            buy_heap = []
            suggestion_heap = []
            sold = False
            logging.info("Sleeping for 60 seconds...")
            time.sleep(60)

        elif status == "early_hours":
            if early_hour_first_iteration:
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    rank = rank_collection.find_one({'strategy': strategy.__name__})['rank']
                    coefficient = r_t_c_collection.find_one({'rank': rank})['coefficient']
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 30 seconds.")
            time.sleep(30)

        elif status == "closed":
            if post_hour_first_iteration:
                early_hour_first_iteration = True
                post_hour_first_iteration = False
                logging.info("Market is closed. Performing post-market operations.")
            time.sleep(30)
        else:
            logging.error("An error occurred while checking market status.")
            time.sleep(60)
    

if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('system.log'),  # Log messages to a file
            logging.StreamHandler()             # Log messages to the console
        ]
    )
    main()

    