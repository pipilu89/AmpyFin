# import sys
# print(sys.version)

# import numpy as np
# import talib

# close = np.random.random(100)

# output = talib.SMA(close)
# print(output)


# import math
# total_portfolio_value = 1000
# trade_asset_limit = 0.115151
# max_investment = total_portfolio_value * trade_asset_limit
# current_price = 244.5
# account_cash = 23

# # shares_based_on_limit = round(max_investment / current_price, 3)
# # shares_based_on_cash = round(account_cash / current_price, 3)
# # shares_based_on_limit2 = math.floor((max_investment / current_price)*100)/100
# # shares_based_on_cash2 = math.floor((account_cash / current_price)*100)/100

# # print(shares_based_on_limit, shares_based_on_cash)
# # print(shares_based_on_limit2, shares_based_on_cash2)
# # print(min(round(max_investment / current_price, 2), round(account_cash / current_price, 2)))

# portfolio_qty = 4.5
# # print(int(portfolio_qty * 0.5))
# print(min(portfolio_qty, max(1, int(portfolio_qty * 0.5))))
# # print(min(portfolio_qty, max(1, (portfolio_qty * 0.5))))

from pymongo import MongoClient
from config import POLYGON_API_KEY, FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, BASE_URL, mongo_url
from bson.decimal128 import Decimal128
import certifi
ca = certifi.where()

# mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
# # Track assets as well
# db = mongo_client.trades
# assets = db.assets_quantities
# limits = db.assets_limit

# symbol ='test'
# qty = 13.251   
# qty2 = 10.69
# assets.update_one({'symbol': symbol}, {'$inc': {'quantity': Decimal128(str(qty))}}, upsert=True)
# assets.update_one({'symbol': symbol}, {'$inc': {'quantity': -qty2}}, upsert=True)

# ticker = symbol
# asset_collection = mongo_client.trades.assets_quantities
# asset_info = asset_collection.find_one({'symbol': ticker})
# portfolio_qty = asset_info['quantity'] if asset_info else 0.0
# # Convert Decimal128 to decimal.Decimal
# portfolio_qty = portfolio_qty.to_decimal()
# # Convert decimal.Decimal to float
# portfolio_qty = float(portfolio_qty)

# print(portfolio_qty)

import talib as ta

def BBANDS_indicator(ticker, data):  
   """Bollinger Bands (BBANDS) indicator."""  
      
   upper, middle, lower = ta.BBANDS(data['Close'], timeperiod=20)  
   if data['Close'].iloc[-1] > upper.iloc[-1]:  
      return 'Sell'  
   elif data['Close'].iloc[-1] < lower.iloc[-1]:  
      return 'Buy'  
   else:  
      return 'Hold' 