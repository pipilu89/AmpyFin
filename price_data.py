import pandas as pd
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from zoneinfo import ZoneInfo
from config import API_KEY, API_SECRET

# setup stock historical data client
stock_historical_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


def get_alpaca_latest_price(ticker_list):
  """
  Fetches the latest price data for a list of tickers from Alpaca API.

  Parameters:
  ticker_list (list): List of ticker symbols to fetch the latest data for.

  Returns:
  dict: A dictionary containing the latest price data for each ticker.
  """
  latest_prices = {}
  request_params = StockSnapshotRequest(
      symbol_or_symbols = ticker_list,
  )
  data = stock_historical_data_client.get_stock_snapshot(request_params)
  for ticker in ticker_list:
    latest_prices[ticker] = data[ticker].minute_bar.close
  return latest_prices

def get_alpaca_historical_price(ticker_list, days):
  """
  Fetches historical price data for a list of tickers from Alpaca API.

  Parameters:
  ticker_list (list): List of ticker symbols to fetch historical data for.
  days (int): Number of days of historical data to fetch.

  Returns:
  pandas.DataFrame: A DataFrame containing the historical price data with columns renamed to:
                    'Open', 'High', 'Low', 'Close', and 'Volume'. (renamed so they are compatible with the TA-Lib library).

  The DataFrame has a MultiIndex with the first level being the ticker symbol and the second level being the timestamp.

  """
  # now = datetime.now(ZoneInfo("America/New_York"))
  now = datetime.now()
  req = StockBarsRequest(
      symbol_or_symbols = ticker_list,
      # timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Hour), # specify timeframe
      timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Day), # specify timeframe
      start = now - timedelta(days = days),    # specify start datetime, default=the beginning of the current day.
      # end_date=None,                        # specify end datetime, default=now
      # limit = 5,                            # specify limit
  )
  df = stock_historical_data_client.get_stock_bars(req).df

  # rename the columns of the dataframe
  df.rename(columns={
      'open': 'Open',
      'high': 'High',
      'low': 'Low',
      'close': 'Close',
      'volume': 'Volume'
  }, inplace=True)
  return df


def filter_df_by_days(df, days, current_date):
    """
    Filters the DataFrame to include only the rows within the specified number of days from the current date.

    Parameters:
    df (pandas.DataFrame): The DataFrame to filter.
    days (int): The number of days to include.
    current_date (datetime): The current date.

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=ZoneInfo("UTC"))
    
    start_date = current_date - timedelta(days=days)
    filtered_df = df[df.index >= start_date]
    return filtered_df




if __name__ == "__main__": 
   
  ticker_list = ['AAPL']
  # ticker_list = ['SPY','AAPL']

  data = get_alpaca_latest_price(ticker_list)
  print(data)
  # for ticker in ticker_list:
  #   print(f"{ticker}: {data[ticker]}")


  df_historical_data = get_alpaca_historical_price(ticker_list, days=120)
  # print(df_historical_data)
  for ticker in ticker_list:
    df_single_ticker_historical_data = df_historical_data.loc[(ticker,)]
    print(df_single_ticker_historical_data)
    print(f"df_single_ticker_historical_data length: {len(df_single_ticker_historical_data)}")


    # Example usage of filter_df_by_days
    # current_date = datetime.now().replace(tzinfo=ZoneInfo("UTC"))
    # days = 3
    # df_filtered = filter_df_by_days(df_single_ticker_historical_data, days, current_date)
    # print(f"Filtered DataFrame length: {len(df_filtered)}")


    import talib as ta

    def HT_TRENDLINE_indicator(ticker, data):  
      """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) indicator."""  
          
      ht_trendline = ta.HT_TRENDLINE(data['Close'])  
      print(ht_trendline)
      if data['Close'].iloc[-1] > ht_trendline.iloc[-1]:  
          return 'Buy'  
      elif data['Close'].iloc[-1] < ht_trendline.iloc[-1]:  
          return 'Sell'  
      else:  
          return 'Hold'
    
    HT_TRENDLINE_indicator(ticker, df_single_ticker_historical_data)