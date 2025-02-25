import pandas as pd
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from zoneinfo import ZoneInfo
from config import API_KEY, API_SECRET
import logging

# setup stock historical data client
stock_historical_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


def get_alpaca_latest_price(ticker_list):
  """
  Fetches the latest price data for a list of tickers from Alpaca API.

  Parameters:
  ticker_list (list): List of ticker symbols to fetch the latest data for.

  Returns:
  df containing the latest price data for each ticker.
  """
  logging.info('download latest price data from alpaca...')
  request_params = StockSnapshotRequest(
      symbol_or_symbols = ticker_list,
  )
  data = stock_historical_data_client.get_stock_snapshot(request_params)

  snapshot_dict_all = []
  for stock, snapshot in data.items():
      snapshot_dict = {
          'Date': snapshot.minute_bar.timestamp,
          'Ticker': stock,
          'Open': snapshot.minute_bar.open,
          'High': snapshot.minute_bar.high,
          'Low': snapshot.minute_bar.low,
          'Close': snapshot.minute_bar.close,
      }
      snapshot_dict_all.append(snapshot_dict)

  # Create DataFrame from list of dictionaries
  snapshot_df = pd.DataFrame(snapshot_dict_all)
  # Set 'Timestamp' as the index
  snapshot_df.set_index('Date', inplace=True)
  logging.info(f"length alpaca latest prices df = {len(snapshot_df)}")
  return snapshot_df
  

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
  logging.info('download historical price data from alpaca...')
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

  # common df format Date(index), Ticker, Open, High, Low, Close, Volume

  # Drop the 'trade_count' and 'vwap' columns
  df.drop(columns=['trade_count', 'vwap'], inplace=True)
  
  # Reset the index to convert the MultiIndex into columns
  df.reset_index(inplace=True)

  # Set the index to the timestamp and rename it to 'Date'
  df.set_index('timestamp', inplace=True)
  df.index.rename('Date', inplace=True)

  # rename the columns of the dataframe
  df.rename(columns={
      'open': 'Open',
      'high': 'High',
      'low': 'Low',
      'close': 'Close',
      'volume': 'Volume',
      'symbol': 'Ticker'
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
   ...