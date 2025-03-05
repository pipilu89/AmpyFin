import pandas as pd
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from zoneinfo import ZoneInfo
from config import API_KEY, API_SECRET
import logging
import os
import yfinance as yf
import time
from helper_files.client_helper import clean_old_files

# setup stock historical data client for alpaca.
stock_historical_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# today_date_str = datetime.now().strftime('%Y-%m-%d')
# historical_data_filename = f'df_historical_prices_{today_date_str}.csv'
# historical_data_directory = '.'  # Directory where the historical data files are stored
# price_data_source = 'yf'  # Source of price data (yf or alpaca)
price_data_source = 'alpaca'  # Source of price data (yf or alpaca)

def get_alpaca_latest_price(ticker_list):
    """
  Fetches the latest price data for a list of tickers from Alpaca API.

  Parameters:
  ticker_list (list): List of ticker symbols to fetch the latest data for.

  Returns:
  df containing the latest price data for each ticker.
  """
    logging.info('download latest price data from alpaca...')
    try:
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
        logging.info(f"finshed getting latest prices from Alpaca. {snapshot_df.shape = }\n{snapshot_df.tail()}")
        # logging.info(f"\n{snapshot_df.tail()}")
        return snapshot_df
    except Exception as e:
        logging.error(f"Error getting latestest prices from alpaca: {e}")
        return pd.DataFrame()
  
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
    try:
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
        logging.info(f"finshed getting historical prices from Alpaca. {len(df) = } {df.shape = }")
        return df
    except Exception as e:
        logging.error(f"Error getting historical prices from alpaca: {e}")
        return pd.DataFrame()

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
        df = pd.DataFrame()
        # Download the latest prices for the tickers
        df = yf.download(tickers, period='1d', interval='1m', group_by='Ticker', auto_adjust=True, repair=True, rounding=True)
        if not df.empty:
            # truncate df to most recent date only
            df = df[df.index == df.index.max()]
            # Stack multi-level column index
            df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            df = df[['Ticker','Open', 'High', 'Low', 'Close', 'Volume']]
            logging.info(f"downloaded latest prices from yfinance. {df.shape = }")
            logging.info(f"\n{df.tail()}")
    except Exception as e:
        logging.error(f"Error downloading latest prices from yfinance: {e}")
    return df
  
def get_yf_historical_data_and_format(ndaq_tickers, period ='2y'):
    """
    Batch download the historical data for multiple tickers using yfinance.
    Parameters:
    - tickers (list): List of ticker symbols.
    """
   # use yf dowloaded data, split period_max('2y') into shorter periods, rather than dl multiple periods. Possible problem if ticker does have 2 years of data (it may have 1mo or 6mo). Test with recent ipo.
#    df = get_historical_data_from_yf(ndaq_tickers, period_max)
    logging.info('download historical price data from yf...')
    try:
        df = yf.download(ndaq_tickers, group_by='Ticker', period=period, interval='1d', auto_adjust=True, repair=True, rounding=True)
        # return df
    except Exception as e:
        logging.error(f'yf historical error: {e}')
        return pd.DataFrame()
    if not df.empty:
        # Stack multi-level column index
        df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        df = df[['Ticker','Open', 'High', 'Low', 'Close', 'Volume']]
        logging.info(f"finshed getting historical prices from yf. {len(df) = } {df.shape = }")
        return df

def adjust_df_length_based_on_period(df, period):
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

def batch_insert_historical_price_df_into_mdb(mongo_client, ndaq_tickers, period_list, df):
   db = mongo_client.HistoricalDatabase
   collection = db.HistoricalDatabase

   logging.info("Deleting historical database...")
   try:
      collection.delete_many({})
      logging.info("Successfully deleted historical database")
   except Exception as e:
      logging.error(f"Error deleting historical database: {e}")
      raise
 
   documents = []
   for period in period_list:
      for ticker in ndaq_tickers:
         df_single_ticker = df.loc[df['Ticker'] == ticker]
         df_single_ticker = df_single_ticker.dropna()

         df_single_ticker = adjust_df_length_based_on_period(df_single_ticker, period)

         records = df_single_ticker.reset_index().to_dict('records')
         documents.append({"ticker": ticker, "period": period, 'data': records})
   if documents:
      try:
        result = collection.insert_many(documents)
        logging.info(f"Historical prices for period {period_list} stored in MongoDB. {result.acknowledged = }. {len(result.inserted_ids) = }")
      except Exception as e:
        logging.error(f"Error storing historical prices in mdb: {e}")
        return
   return result.acknowledged

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

def get_historical_prices(mongo_client, ndaq_tickers, period_list, max_retries=5, initial_delay=60):
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    historical_data_filename = f'df_historical_prices_{today_date_str}.csv'
    historical_data_directory = '.'  # Directory where the historical data files are stored           
    if not os.path.exists(historical_data_filename):
        
        df_historical_prices = pd.DataFrame()
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            if price_data_source == 'yf':
                df_historical_prices = get_yf_historical_data_and_format(ndaq_tickers, period ='2y')
            elif price_data_source == 'alpaca':
                df_historical_prices = get_alpaca_historical_price(ndaq_tickers, 730)
        
            if not df_historical_prices.empty:
                # store df_historical_yf_prices in local file
                df_historical_prices.to_csv(historical_data_filename, index=True)
                logging.info(f"Downloaded and saved df_historical_prices to {historical_data_filename}")
                # Clean old files to maintain a maximum of 5 files
                clean_old_files(historical_data_directory, 'df_historical_prices_*.csv', 5)
                #store in mdb
                response = batch_insert_historical_price_df_into_mdb(mongo_client, ndaq_tickers, period_list, df_historical_prices)
                return df_historical_prices
            
            retries += 1
            logging.warning(f"Getting latest prices attempt {retries} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff

        logging.error("Getting historical prices. Max retries reached. Failed to get historical prices.")
        return pd.DataFrame()

    else:
        # load df_historical_prices from local file
        try:
            df_historical_prices = pd.read_csv(historical_data_filename)
            logging.info(f"Loaded df_historical_prices from {historical_data_filename}. {df_historical_prices.shape = }")
        except Exception as e:
            logging.error(f"Error loading {historical_data_filename}: {e}")
    logging.info(f"\n{df_historical_prices.head()}")
    logging.info(f"\n{df_historical_prices.tail()}")
    return df_historical_prices

def get_latest_prices(ndaq_tickers, max_retries=5, initial_delay=60):
    """
    Fetches the latest prices for the given tickers with exponential backoff retry if the DataFrame is empty.

    Parameters:
    - ndaq_tickers (list): List of ticker symbols.
    - max_retries (int): Maximum number of retries.
    - initial_delay (int): Initial delay in seconds before retrying.

    Returns:
    - pandas.DataFrame: DataFrame containing the latest prices.
    """
    try:
        df_latest_prices = pd.DataFrame()
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            if price_data_source == 'yf':
                df_latest_prices = get_latest_prices_from_yf2(ndaq_tickers)
            elif price_data_source == 'alpaca':
                df_latest_prices = get_alpaca_latest_price(ndaq_tickers)

            if not df_latest_prices.empty:
                # save df to local file
                df_latest_prices.to_csv('latest_prices.csv', index=True)
                return df_latest_prices

            retries += 1
            logging.warning(f"Getting latest prices attempt {retries} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff

        logging.error("Getting latest prices. Max retries reached. Failed to get latest prices.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error getting latest prices: {e}")
        return pd.DataFrame()

if __name__ == "__main__": 
   ...