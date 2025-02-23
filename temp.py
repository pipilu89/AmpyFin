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

# action_talib_dict = {} #talib indicator results
# ndaq_tickers = ["AAPL", "AMD", "GOOG"]


# def main():
#   global action_talib_dict
#   for ticker in ndaq_tickers:
#     action_talib_dict = {ticker:{}} #talib indicator results

# main()

# print(action_talib_dict)


action_talib_dict = {
  'AAPL': 
          {'BBANDS_indicator': 'Hold', 'DEMA_indicator': 'Hold', 'EMA_indicator': 'Buy', 'HT_TRENDLINE_indicator': 'Buy', 'KAMA_indicator': 'Hold', 'MA_indicator': 'Buy', 'MAMA_indicator': 'Buy', 'MAVP_indicator': 'Buy', 'MIDPOINT_indicator': 'Buy', 'MIDPRICE_indicator': 'Buy', 'SAR_indicator': 'Buy', 'SAREXT_indicator': 'Buy', 'SMA_indicator': 'Buy', 'T3_indicator': 'Hold', 'TEMA_indicator': 'Hold', 'TRIMA_indicator': 'Buy', 'WMA_indicator': 'Buy', 'ADX_indicator': 'Hold', 'ADXR_indicator': 'Hold', 'APO_indicator': 'Buy', 'AROON_indicator': 'Buy', 'AROONOSC_indicator': 'Buy', 'BOP_indicator': 'Sell', 'CCI_indicator': 'Buy', 'CMO_indicator': 'Hold', 'DX_indicator': 'Buy', 'MACD_indicator': 'Buy', 'MACDEXT_indicator': 'Buy', 'MACDFIX_indicator': 'Buy', 'MFI_indicator': 'Hold', 'MINUS_DI_indicator': 'Buy', 'MINUS_DM_indicator': 'Sell', 'MOM_indicator': 'Buy', 'PLUS_DI_indicator': 'Buy', 'PLUS_DM_indicator': 'Buy', 
'PPO_indicator': 'Buy', 'ROC_indicator': 'Buy', 'ROCP_indicator': 'Buy', 'ROCR_indicator': 'Buy', 'ROCR100_indicator': 'Buy', 'RSI_indicator': 'Hold', 'STOCH_indicator': 'Sell', 'STOCHF_indicator': 'Hold', 'STOCHRSI_indicator': 'Hold', 'TRIX_indicator': 'Hold', 'ULTOSC_indicator': 'Hold', 'WILLR_indicator': 'Sell', 'AD_indicator': 'Buy', 'ADOSC_indicator': 'Buy', 'OBV_indicator': 'Sell', 'HT_DCPERIOD_indicator': 'Buy', 'HT_DCPHASE_indicator': 'Buy', 'HT_PHASOR_indicator': 'Buy', 'HT_SINE_indicator': 'Buy', 'HT_TRENDMODE_indicator': 'Buy', 'AVGPRICE_indicator': 'Sell', 'MEDPRICE_indicator': 'Sell', 'TYPPRICE_indicator': 'Sell', 'WCLPRICE_indicator': 'Sell', 'ATR_indicator': 'Sell', 'NATR_indicator': 'Sell', 'TRANGE_indicator': 'Sell', 'CDL2CROWS_indicator': 'Hold', 'CDL3BLACKCROWS_indicator': 'Hold', 'CDL3INSIDE_indicator': 'Hold', 'CDL3LINESTRIKE_indicator': 'Hold', 'CDL3OUTSIDE_indicator': 'Hold', 'CDL3STARSINSOUTH_indicator': 'Hold', 'CDL3WHITESOLDIERS_indicator': 'Hold', 'CDLABANDONEDBABY_indicator': 'Hold', 'CDLADVANCEBLOCK_indicator': 'Hold', 'CDLBELTHOLD_indicator': 'Hold', 'CDLBREAKAWAY_indicator': 'Hold', 'CDLCLOSINGMARUBOZU_indicator': 'Hold', 'CDLCONCEALBABYSWALL_indicator': 'Hold', 'CDLCOUNTERATTACK_indicator': 'Hold', 
'CDLDARKCLOUDCOVER_indicator': 'Hold', 'CDLDOJI_indicator': 'Buy', 'CDLDOJISTAR_indicator': 'Hold', 'CDLDRAGONFLYDOJI_indicator': 'Hold', 'CDLENGULFING_indicator': 'Hold', 'CDLEVENINGDOJISTAR_indicator': 'Hold', 'CDLEVENINGSTAR_indicator': 'Hold', 'CDLGAPSIDESIDEWHITE_indicator': 'Hold', 'CDLGRAVESTONEDOJI_indicator': 'Buy', 'CDLHAMMER_indicator': 'Hold', 'CDLHANGINGMAN_indicator': 'Hold', 'CDLHARAMI_indicator': 'Hold', 'CDLHARAMICROSS_indicator': 'Hold', 'CDLHIGHWAVE_indicator': 'Hold', 'CDLHIKKAKE_indicator': 'Hold', 'CDLHIKKAKEMOD_indicator': 'Hold', 'CDLHOMINGPIGEON_indicator': 'Hold', 'CDLIDENTICAL3CROWS_indicator': 'Hold', 'CDLINNECK_indicator': 'Hold', 'CDLINVERTEDHAMMER_indicator': 'Hold', 'CDLKICKING_indicator': 'Hold', 'CDLKICKINGBYLENGTH_indicator': 'Hold', 'CDLLADDERBOTTOM_indicator': 'Hold', 'CDLLONGLEGGEDDOJI_indicator': 'Buy', 'CDLLONGLINE_indicator': 'Hold', 'CDLMARUBOZU_indicator': 'Hold', 'CDLMATCHINGLOW_indicator': 'Hold', 'CDLMATHOLD_indicator': 'Hold', 'CDLMORNINGDOJISTAR_indicator': 'Hold', 'CDLMORNINGSTAR_indicator': 'Hold', 'CDLONNECK_indicator': 'Hold', 'CDLPIERCING_indicator': 'Hold', 'CDLRICKSHAWMAN_indicator': 'Hold', 'CDLRISEFALL3METHODS_indicator': 'Hold', 'CDLSEPARATINGLINES_indicator': 'Hold', 'CDLSHOOTINGSTAR_indicator': 'Hold', 'CDLSHORTLINE_indicator': 'Hold', 'CDLSPINNINGTOP_indicator': 'Hold', 'CDLSTALLEDPATTERN_indicator': 'Hold', 'CDLSTICKSANDWICH_indicator': 'Hold', 'CDLTAKURI_indicator': 'Hold', 'CDLTASUKIGAP_indicator': 'Hold', 'CDLTHRUSTING_indicator': 'Hold', 'CDLTRISTAR_indicator': 'Hold', 'CDLUNIQUE3RIVER_indicator': 'Hold', 'CDLUPSIDEGAP2CROWS_indicator': 'Hold', 'CDLXSIDEGAP3METHODS_indicator': 'Hold', 'BETA_indicator': 'Buy', 'CORREL_indicator': 'Buy', 'LINEARREG_indicator': 'Sell', 'LINEARREG_ANGLE_indicator': 'Buy', 'LINEARREG_INTERCEPT_indicator': 'Buy', 'LINEARREG_SLOPE_indicator': 'Buy', 'STDDEV_indicator': 'Sell', 'TSF_indicator': 'Sell', 'VAR_indicator': 'Sell'}, 
'AMD': 
{'BBANDS_indicator': 'Hold', 'DEMA_indicator': 'Hold', 'EMA_indicator': 'Sell', 'HT_TRENDLINE_indicator': 'Sell', 'KAMA_indicator': 'Hold', 'MA_indicator': 'Sell', 'MAMA_indicator': 'Sell', 'MAVP_indicator': 'Sell', 'MIDPOINT_indicator': 'Sell', 'MIDPRICE_indicator': 'Sell', 'SAR_indicator': 'Sell', 'SAREXT_indicator': 'Buy', 'SMA_indicator': 'Sell', 'T3_indicator': 'Hold', 'TEMA_indicator': 'Hold', 'TRIMA_indicator': 'Sell', 'WMA_indicator': 'Sell', 'ADX_indicator': 'Sell', 'ADXR_indicator': 'Sell', 'APO_indicator': 'Sell', 'AROON_indicator': 'Hold', 'AROONOSC_indicator': 'Sell', 'BOP_indicator': 'Sell', 'CCI_indicator': 'Hold', 'CMO_indicator': 'Hold', 'DX_indicator': 'Sell', 'MACD_indicator': 'Buy', 'MACDEXT_indicator': 'Sell', 'MACDFIX_indicator': 'Buy', 'MFI_indicator': 'Hold', 'MINUS_DI_indicator': 'Hold', 'MINUS_DM_indicator': 'Sell', 'MOM_indicator': 'Buy', 'PLUS_DI_indicator': 'Hold', 'PLUS_DM_indicator': 'Buy', 'PPO_indicator': 'Sell', 'ROC_indicator': 'Buy', 'ROCP_indicator': 'Buy', 'ROCR_indicator': 'Buy', 'ROCR100_indicator': 'Buy', 'RSI_indicator': 'Hold', 'STOCH_indicator': 'Hold', 'STOCHF_indicator': 'Buy', 'STOCHRSI_indicator': 'Buy', 'TRIX_indicator': 'Hold', 'ULTOSC_indicator': 'Hold', 'WILLR_indicator': 'Hold', 'AD_indicator': 'Buy', 'ADOSC_indicator': 'Sell', 'OBV_indicator': 'Sell', 'HT_DCPERIOD_indicator': 'Hold', 'HT_DCPHASE_indicator': 'Buy', 'HT_PHASOR_indicator': 'Buy', 'HT_SINE_indicator': 'Buy', 'HT_TRENDMODE_indicator': 'Hold', 'AVGPRICE_indicator': 'Sell', 'MEDPRICE_indicator': 'Sell', 'TYPPRICE_indicator': 'Sell', 'WCLPRICE_indicator': 'Sell', 'ATR_indicator': 'Sell', 'NATR_indicator': 'Sell', 'TRANGE_indicator': 'Sell', 'CDL2CROWS_indicator': 'Hold', 'CDL3BLACKCROWS_indicator': 'Hold', 'CDL3INSIDE_indicator': 'Hold', 'CDL3LINESTRIKE_indicator': 'Hold', 'CDL3OUTSIDE_indicator': 'Sell', 'CDL3STARSINSOUTH_indicator': 'Hold', 'CDL3WHITESOLDIERS_indicator': 'Hold', 'CDLABANDONEDBABY_indicator': 'Hold', 'CDLADVANCEBLOCK_indicator': 'Hold', 'CDLBELTHOLD_indicator': 'Hold', 'CDLBREAKAWAY_indicator': 'Hold', 'CDLCLOSINGMARUBOZU_indicator': 'Hold', 'CDLCONCEALBABYSWALL_indicator': 'Hold', 'CDLCOUNTERATTACK_indicator': 'Hold', 'CDLDARKCLOUDCOVER_indicator': 'Hold', 'CDLDOJI_indicator': 'Hold', 'CDLDOJISTAR_indicator': 'Hold', 'CDLDRAGONFLYDOJI_indicator': 'Hold', 'CDLENGULFING_indicator': 'Hold', 'CDLEVENINGDOJISTAR_indicator': 'Hold', 'CDLEVENINGSTAR_indicator': 'Hold', 'CDLGAPSIDESIDEWHITE_indicator': 'Hold', 'CDLGRAVESTONEDOJI_indicator': 'Hold', 'CDLHAMMER_indicator': 'Hold', 'CDLHANGINGMAN_indicator': 'Hold', 'CDLHARAMI_indicator': 'Hold', 'CDLHARAMICROSS_indicator': 'Hold', 'CDLHIGHWAVE_indicator': 'Hold', 'CDLHIKKAKE_indicator': 'Hold', 'CDLHIKKAKEMOD_indicator': 'Hold', 'CDLHOMINGPIGEON_indicator': 'Hold', 'CDLIDENTICAL3CROWS_indicator': 'Hold', 'CDLINNECK_indicator': 'Hold', 'CDLINVERTEDHAMMER_indicator': 'Hold', 'CDLKICKING_indicator': 'Hold', 'CDLKICKINGBYLENGTH_indicator': 'Hold', 'CDLLADDERBOTTOM_indicator': 'Hold', 'CDLLONGLEGGEDDOJI_indicator': 'Hold', 'CDLLONGLINE_indicator': 'Sell', 'CDLMARUBOZU_indicator': 'Hold', 'CDLMATCHINGLOW_indicator': 'Hold', 'CDLMATHOLD_indicator': 'Hold', 'CDLMORNINGDOJISTAR_indicator': 'Hold', 'CDLMORNINGSTAR_indicator': 'Hold', 'CDLONNECK_indicator': 'Hold', 'CDLPIERCING_indicator': 'Hold', 'CDLRICKSHAWMAN_indicator': 'Hold', 'CDLRISEFALL3METHODS_indicator': 'Hold', 'CDLSEPARATINGLINES_indicator': 'Hold', 'CDLSHOOTINGSTAR_indicator': 'Hold', 'CDLSHORTLINE_indicator': 'Hold', 'CDLSPINNINGTOP_indicator': 'Hold', 'CDLSTALLEDPATTERN_indicator': 'Hold', 'CDLSTICKSANDWICH_indicator': 'Hold', 'CDLTAKURI_indicator': 'Hold', 'CDLTASUKIGAP_indicator': 'Hold', 'CDLTHRUSTING_indicator': 'Hold', 'CDLTRISTAR_indicator': 'Hold', 'CDLUNIQUE3RIVER_indicator': 'Hold', 'CDLUPSIDEGAP2CROWS_indicator': 'Hold', 'CDLXSIDEGAP3METHODS_indicator': 'Hold', 'BETA_indicator': 'Buy', 'CORREL_indicator': 'Buy', 'LINEARREG_indicator': 'Sell', 'LINEARREG_ANGLE_indicator': 'Sell', 'LINEARREG_INTERCEPT_indicator': 'Sell', 'LINEARREG_SLOPE_indicator': 'Sell', 'STDDEV_indicator': 'Sell', 'TSF_indicator': 'Sell', 'VAR_indicator': 'Sell'}, 
'GOOG': 
{'BBANDS_indicator': 'Hold', 'DEMA_indicator': 'Hold', 'EMA_indicator': 'Sell', 'HT_TRENDLINE_indicator': 'Sell', 'KAMA_indicator': 'Hold', 'MA_indicator': 'Sell', 'MAMA_indicator': 'Sell', 'MAVP_indicator': 'Sell', 'MIDPOINT_indicator': 'Sell', 'MIDPRICE_indicator': 'Sell', 'SAR_indicator': 'Buy', 'SAREXT_indicator': 'Buy', 'SMA_indicator': 'Sell', 'T3_indicator': 'Hold', 'TEMA_indicator': 'Hold', 'TRIMA_indicator': 'Sell', 'WMA_indicator': 'Sell', 'ADX_indicator': 'Hold', 'ADXR_indicator': 'Hold', 'APO_indicator': 'Sell', 'AROON_indicator': 'Sell', 'AROONOSC_indicator': 'Sell', 'BOP_indicator': 'Sell', 'CCI_indicator': 'Hold', 'CMO_indicator': 'Hold', 'DX_indicator': 'Buy', 'MACD_indicator': 'Sell', 'MACDEXT_indicator': 'Sell', 'MACDFIX_indicator': 'Sell', 'MFI_indicator': 'Hold', 'MINUS_DI_indicator': 'Sell', 'MINUS_DM_indicator': 'Sell', 'MOM_indicator': 'Sell', 'PLUS_DI_indicator': 'Sell', 'PLUS_DM_indicator': 'Buy', 'PPO_indicator': 'Sell', 'ROC_indicator': 'Sell', 'ROCP_indicator': 'Sell', 'ROCR_indicator': 'Sell', 'ROCR100_indicator': 'Sell', 'RSI_indicator': 'Hold', 'STOCH_indicator': 'Hold', 'STOCHF_indicator': 'Buy', 'STOCHRSI_indicator': 'Buy', 'TRIX_indicator': 'Hold', 'ULTOSC_indicator': 'Hold', 'WILLR_indicator': 'Buy', 'AD_indicator': 'Buy', 'ADOSC_indicator': 'Buy', 'OBV_indicator': 'Sell', 'HT_DCPERIOD_indicator': 'Buy', 'HT_DCPHASE_indicator': 'Sell', 'HT_PHASOR_indicator': 'Sell', 'HT_SINE_indicator': 'Sell', 'HT_TRENDMODE_indicator': 'Buy', 'AVGPRICE_indicator': 'Sell', 'MEDPRICE_indicator': 'Sell', 'TYPPRICE_indicator': 'Sell', 'WCLPRICE_indicator': 'Sell', 'ATR_indicator': 'Sell', 'NATR_indicator': 'Sell', 'TRANGE_indicator': 'Sell', 'CDL2CROWS_indicator': 'Hold', 'CDL3BLACKCROWS_indicator': 'Hold', 'CDL3INSIDE_indicator': 'Hold', 'CDL3LINESTRIKE_indicator': 'Hold', 'CDL3OUTSIDE_indicator': 'Hold', 'CDL3STARSINSOUTH_indicator': 'Hold', 'CDL3WHITESOLDIERS_indicator': 'Hold', 'CDLABANDONEDBABY_indicator': 'Hold', 'CDLADVANCEBLOCK_indicator': 'Hold', 'CDLBELTHOLD_indicator': 'Sell', 'CDLBREAKAWAY_indicator': 'Hold', 'CDLCLOSINGMARUBOZU_indicator': 'Hold', 'CDLCONCEALBABYSWALL_indicator': 'Hold', 'CDLCOUNTERATTACK_indicator': 'Hold', 'CDLDARKCLOUDCOVER_indicator': 'Hold', 'CDLDOJI_indicator': 'Hold', 'CDLDOJISTAR_indicator': 'Hold', 'CDLDRAGONFLYDOJI_indicator': 'Hold', 'CDLENGULFING_indicator': 'Sell', 'CDLEVENINGDOJISTAR_indicator': 'Hold', 'CDLEVENINGSTAR_indicator': 'Hold', 'CDLGAPSIDESIDEWHITE_indicator': 'Hold', 'CDLGRAVESTONEDOJI_indicator': 'Hold', 'CDLHAMMER_indicator': 'Hold', 'CDLHANGINGMAN_indicator': 'Hold', 'CDLHARAMI_indicator': 'Hold', 'CDLHARAMICROSS_indicator': 'Hold', 'CDLHIGHWAVE_indicator': 'Hold', 'CDLHIKKAKE_indicator': 'Hold', 'CDLHIKKAKEMOD_indicator': 'Hold', 'CDLHOMINGPIGEON_indicator': 'Hold', 'CDLIDENTICAL3CROWS_indicator': 'Hold', 'CDLINNECK_indicator': 'Hold', 'CDLINVERTEDHAMMER_indicator': 'Hold', 'CDLKICKING_indicator': 'Hold', 'CDLKICKINGBYLENGTH_indicator': 'Hold', 'CDLLADDERBOTTOM_indicator': 'Hold', 'CDLLONGLEGGEDDOJI_indicator': 'Hold', 'CDLLONGLINE_indicator': 'Sell', 'CDLMARUBOZU_indicator': 'Hold', 'CDLMATCHINGLOW_indicator': 'Hold', 'CDLMATHOLD_indicator': 'Hold', 'CDLMORNINGDOJISTAR_indicator': 'Hold', 'CDLMORNINGSTAR_indicator': 'Hold', 'CDLONNECK_indicator': 'Hold', 'CDLPIERCING_indicator': 'Hold', 'CDLRICKSHAWMAN_indicator': 'Hold', 'CDLRISEFALL3METHODS_indicator': 'Hold', 'CDLSEPARATINGLINES_indicator': 'Hold', 'CDLSHOOTINGSTAR_indicator': 'Hold', 'CDLSHORTLINE_indicator': 'Hold', 'CDLSPINNINGTOP_indicator': 'Hold', 'CDLSTALLEDPATTERN_indicator': 'Hold', 'CDLSTICKSANDWICH_indicator': 'Hold', 'CDLTAKURI_indicator': 'Hold', 'CDLTASUKIGAP_indicator': 'Hold', 'CDLTHRUSTING_indicator': 'Hold', 'CDLTRISTAR_indicator': 'Hold', 'CDLUNIQUE3RIVER_indicator': 'Hold', 'CDLUPSIDEGAP2CROWS_indicator': 'Hold', 'CDLXSIDEGAP3METHODS_indicator': 'Hold', 'BETA_indicator': 'Sell', 'CORREL_indicator': 'Buy', 'LINEARREG_indicator': 'Buy', 'LINEARREG_ANGLE_indicator': 'Sell', 'LINEARREG_INTERCEPT_indicator': 'Sell', 'LINEARREG_SLOPE_indicator': 'Sell', 'STDDEV_indicator': 'Sell', 'TSF_indicator': 'Buy', 'VAR_indicator': 'Sell'}}

# Dictionary to store the summed actions
def summarize_action_talib_dict(action_talib_dict):
    summary = {}
    for ticker, indicators in action_talib_dict.items():
        summary[ticker] = {'Buy': 0, 'Sell': 0, 'Hold': 0}
        for action in indicators.values():
            if action in summary[ticker]:
                summary[ticker][action] += 1

        summary[ticker]["total"] = sum(summary[ticker].values())
        print(f"{ticker}: {summary[ticker]}")
    return summary

# Example usage
summary = summarize_action_talib_dict(action_talib_dict)
print(summary)

def summarize_by_indicator(action_talib_dict):
    summary = {}
    for indicators in action_talib_dict.values():
        for indicator, action in indicators.items():
            if indicator not in summary:
                summary[indicator] = {'Buy': 0, 'Sell': 0, 'Hold': 0}
            if action in summary[indicator]:
                summary[indicator][action] += 1
    return summary

# Example usage
# indicator_summary = summarize_by_indicator(action_talib_dict)
# print(indicator_summary)