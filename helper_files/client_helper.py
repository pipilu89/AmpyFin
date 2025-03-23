import json
import os
import pandas as pd
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import yfinance as yf
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from pymongo import MongoClient

from control import stop_loss, take_profit
from strategies.talib_indicators import (
    AD_indicator,
    ADOSC_indicator,
    ADX_indicator,
    ADXR_indicator,
    APO_indicator,
    AROON_indicator,
    AROONOSC_indicator,
    ATR_indicator,
    AVGPRICE_indicator,
    BBANDS_indicator,
    BETA_indicator,
    BOP_indicator,
    CCI_indicator,
    CDL2CROWS_indicator,
    CDL3BLACKCROWS_indicator,
    CDL3INSIDE_indicator,
    CDL3LINESTRIKE_indicator,
    CDL3OUTSIDE_indicator,
    CDL3STARSINSOUTH_indicator,
    CDL3WHITESOLDIERS_indicator,
    CDLABANDONEDBABY_indicator,
    CDLADVANCEBLOCK_indicator,
    CDLBELTHOLD_indicator,
    CDLBREAKAWAY_indicator,
    CDLCLOSINGMARUBOZU_indicator,
    CDLCONCEALBABYSWALL_indicator,
    CDLCOUNTERATTACK_indicator,
    CDLDARKCLOUDCOVER_indicator,
    CDLDOJI_indicator,
    CDLDOJISTAR_indicator,
    CDLDRAGONFLYDOJI_indicator,
    CDLENGULFING_indicator,
    CDLEVENINGDOJISTAR_indicator,
    CDLEVENINGSTAR_indicator,
    CDLGAPSIDESIDEWHITE_indicator,
    CDLGRAVESTONEDOJI_indicator,
    CDLHAMMER_indicator,
    CDLHANGINGMAN_indicator,
    CDLHARAMI_indicator,
    CDLHARAMICROSS_indicator,
    CDLHIGHWAVE_indicator,
    CDLHIKKAKE_indicator,
    CDLHIKKAKEMOD_indicator,
    CDLHOMINGPIGEON_indicator,
    CDLIDENTICAL3CROWS_indicator,
    CDLINNECK_indicator,
    CDLINVERTEDHAMMER_indicator,
    CDLKICKING_indicator,
    CDLKICKINGBYLENGTH_indicator,
    CDLLADDERBOTTOM_indicator,
    CDLLONGLEGGEDDOJI_indicator,
    CDLLONGLINE_indicator,
    CDLMARUBOZU_indicator,
    CDLMATCHINGLOW_indicator,
    CDLMATHOLD_indicator,
    CDLMORNINGDOJISTAR_indicator,
    CDLMORNINGSTAR_indicator,
    CDLONNECK_indicator,
    CDLPIERCING_indicator,
    CDLRICKSHAWMAN_indicator,
    CDLRISEFALL3METHODS_indicator,
    CDLSEPARATINGLINES_indicator,
    CDLSHOOTINGSTAR_indicator,
    CDLSHORTLINE_indicator,
    CDLSPINNINGTOP_indicator,
    CDLSTALLEDPATTERN_indicator,
    CDLSTICKSANDWICH_indicator,
    CDLTAKURI_indicator,
    CDLTASUKIGAP_indicator,
    CDLTHRUSTING_indicator,
    CDLTRISTAR_indicator,
    CDLUNIQUE3RIVER_indicator,
    CDLUPSIDEGAP2CROWS_indicator,
    CDLXSIDEGAP3METHODS_indicator,
    CMO_indicator,
    CORREL_indicator,
    DEMA_indicator,
    DX_indicator,
    EMA_indicator,
    HT_DCPERIOD_indicator,
    HT_DCPHASE_indicator,
    HT_PHASOR_indicator,
    HT_SINE_indicator,
    HT_TRENDLINE_indicator,
    HT_TRENDMODE_indicator,
    KAMA_indicator,
    LINEARREG_ANGLE_indicator,
    LINEARREG_indicator,
    LINEARREG_INTERCEPT_indicator,
    LINEARREG_SLOPE_indicator,
    MA_indicator,
    MACD_indicator,
    MACDEXT_indicator,
    MACDFIX_indicator,
    MAMA_indicator,
    MAVP_indicator,
    MEDPRICE_indicator,
    MFI_indicator,
    MIDPOINT_indicator,
    MIDPRICE_indicator,
    MINUS_DI_indicator,
    MINUS_DM_indicator,
    MOM_indicator,
    NATR_indicator,
    OBV_indicator,
    PLUS_DI_indicator,
    PLUS_DM_indicator,
    PPO_indicator,
    ROC_indicator,
    ROCP_indicator,
    ROCR100_indicator,
    ROCR_indicator,
    RSI_indicator,
    SAR_indicator,
    SAREXT_indicator,
    SMA_indicator,
    STDDEV_indicator,
    STOCH_indicator,
    STOCHF_indicator,
    STOCHRSI_indicator,
    T3_indicator,
    TEMA_indicator,
    TRANGE_indicator,
    TRIMA_indicator,
    TRIX_indicator,
    TSF_indicator,
    TYPPRICE_indicator,
    ULTOSC_indicator,
    VAR_indicator,
    WCLPRICE_indicator,
    WILLR_indicator,
    WMA_indicator,
)

sys.path.append("..")

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


overlap_studies = [
    BBANDS_indicator,
    DEMA_indicator,
    EMA_indicator,
    HT_TRENDLINE_indicator,
    KAMA_indicator,
    MA_indicator,
    MAMA_indicator,
    MAVP_indicator,
    MIDPOINT_indicator,
    MIDPRICE_indicator,
    SAR_indicator,
    SAREXT_indicator,
    SMA_indicator,
    T3_indicator,
    TEMA_indicator,
    TRIMA_indicator,
    WMA_indicator,
]
momentum_indicators = [
    ADX_indicator,
    ADXR_indicator,
    APO_indicator,
    AROON_indicator,
    AROONOSC_indicator,
    BOP_indicator,
    CCI_indicator,
    CMO_indicator,
    DX_indicator,
    MACD_indicator,
    MACDEXT_indicator,
    MACDFIX_indicator,
    MFI_indicator,
    MINUS_DI_indicator,
    MINUS_DM_indicator,
    MOM_indicator,
    PLUS_DI_indicator,
    PLUS_DM_indicator,
    PPO_indicator,
    ROC_indicator,
    ROCP_indicator,
    ROCR_indicator,
    ROCR100_indicator,
    RSI_indicator,
    STOCH_indicator,
    STOCHF_indicator,
    STOCHRSI_indicator,
    TRIX_indicator,
    ULTOSC_indicator,
    WILLR_indicator,
]
volume_indicators = [AD_indicator, ADOSC_indicator, OBV_indicator]
cycle_indicators = [
    HT_DCPERIOD_indicator,
    HT_DCPHASE_indicator,
    HT_PHASOR_indicator,
    HT_SINE_indicator,
    HT_TRENDMODE_indicator,
]
price_transforms = [
    AVGPRICE_indicator,
    MEDPRICE_indicator,
    TYPPRICE_indicator,
    WCLPRICE_indicator,
]
volatility_indicators = [ATR_indicator, NATR_indicator, TRANGE_indicator]
pattern_recognition = [
    CDL2CROWS_indicator,
    CDL3BLACKCROWS_indicator,
    CDL3INSIDE_indicator,
    CDL3LINESTRIKE_indicator,
    CDL3OUTSIDE_indicator,
    CDL3STARSINSOUTH_indicator,
    CDL3WHITESOLDIERS_indicator,
    CDLABANDONEDBABY_indicator,
    CDLADVANCEBLOCK_indicator,
    CDLBELTHOLD_indicator,
    CDLBREAKAWAY_indicator,
    CDLCLOSINGMARUBOZU_indicator,
    CDLCONCEALBABYSWALL_indicator,
    CDLCOUNTERATTACK_indicator,
    CDLDARKCLOUDCOVER_indicator,
    CDLDOJI_indicator,
    CDLDOJISTAR_indicator,
    CDLDRAGONFLYDOJI_indicator,
    CDLENGULFING_indicator,
    CDLEVENINGDOJISTAR_indicator,
    CDLEVENINGSTAR_indicator,
    CDLGAPSIDESIDEWHITE_indicator,
    CDLGRAVESTONEDOJI_indicator,
    CDLHAMMER_indicator,
    CDLHANGINGMAN_indicator,
    CDLHARAMI_indicator,
    CDLHARAMICROSS_indicator,
    CDLHIGHWAVE_indicator,
    CDLHIKKAKE_indicator,
    CDLHIKKAKEMOD_indicator,
    CDLHOMINGPIGEON_indicator,
    CDLIDENTICAL3CROWS_indicator,
    CDLINNECK_indicator,
    CDLINVERTEDHAMMER_indicator,
    CDLKICKING_indicator,
    CDLKICKINGBYLENGTH_indicator,
    CDLLADDERBOTTOM_indicator,
    CDLLONGLEGGEDDOJI_indicator,
    CDLLONGLINE_indicator,
    CDLMARUBOZU_indicator,
    CDLMATCHINGLOW_indicator,
    CDLMATHOLD_indicator,
    CDLMORNINGDOJISTAR_indicator,
    CDLMORNINGSTAR_indicator,
    CDLONNECK_indicator,
    CDLPIERCING_indicator,
    CDLRICKSHAWMAN_indicator,
    CDLRISEFALL3METHODS_indicator,
    CDLSEPARATINGLINES_indicator,
    CDLSHOOTINGSTAR_indicator,
    CDLSHORTLINE_indicator,
    CDLSPINNINGTOP_indicator,
    CDLSTALLEDPATTERN_indicator,
    CDLSTICKSANDWICH_indicator,
    CDLTAKURI_indicator,
    CDLTASUKIGAP_indicator,
    CDLTHRUSTING_indicator,
    CDLTRISTAR_indicator,
    CDLUNIQUE3RIVER_indicator,
    CDLUPSIDEGAP2CROWS_indicator,
    CDLXSIDEGAP3METHODS_indicator,
]
statistical_functions = [
    BETA_indicator,
    CORREL_indicator,
    LINEARREG_indicator,
    LINEARREG_ANGLE_indicator,
    LINEARREG_INTERCEPT_indicator,
    LINEARREG_SLOPE_indicator,
    STDDEV_indicator,
    TSF_indicator,
    VAR_indicator,
]

strategies = (
    overlap_studies
    + momentum_indicators
    + volume_indicators
    + cycle_indicators
    + price_transforms
    + volatility_indicators
    + pattern_recognition
    + statistical_functions
)

strategies_test = (
    overlap_studies
    # + momentum_indicators
    # + volume_indicators
    # + cycle_indicators
    # + price_transforms
    # + volatility_indicators
    # + pattern_recognition
    # + statistical_functions
)


# MongoDB connection helper
def connect_to_mongo(mongo_url):
    """Connect to MongoDB and return the client."""
    return MongoClient(mongo_url)


# Helper to place an order
def place_order(trading_client, symbol, side, quantity, mongo_client):
    """
    Place a market order and log the order to MongoDB.

    :param trading_client: The Alpaca trading client instance
    :param symbol: The stock symbol to trade
    :param side: Order side (OrderSide.BUY or OrderSide.SELL)
    :param qty: Quantity to trade
    :param mongo_client: MongoDB client instance
    :return: Order result from Alpaca API
    """

    market_order_data = MarketOrderRequest(
        symbol=symbol, qty=quantity, side=side, time_in_force=TimeInForce.DAY
    )
    order = trading_client.submit_order(market_order_data)
    qty = round(quantity, 3)
    current_price = get_latest_price(symbol)
    stop_loss_price = round(current_price * (1 - stop_loss), 2)  # 3% loss
    take_profit_price = round(current_price * (1 + take_profit), 2)  # 5% profit

    # Log trade details to MongoDB
    db = mongo_client.trades
    db.paper.insert_one(
        {
            "symbol": symbol,
            "qty": qty,
            "side": side.name,
            "time_in_force": TimeInForce.DAY.name,
            "time": datetime.now(tz=timezone.utc),
        }
    )

    # Track assets as well
    assets = db.assets_quantities
    limits = db.assets_limit

    if side == OrderSide.BUY:
        assets.update_one({"symbol": symbol}, {"$inc": {"quantity": qty}}, upsert=True)
        limits.update_one(
            {"symbol": symbol},
            {
                "$set": {
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                }
            },
            upsert=True,
        )
    elif side == OrderSide.SELL:
        assets.update_one({"symbol": symbol}, {"$inc": {"quantity": -qty}}, upsert=True)
        if assets.find_one({"symbol": symbol})["quantity"] == 0:
            assets.delete_one({"symbol": symbol})
            limits.delete_one({"symbol": symbol})

    return order


# Helper to retrieve NASDAQ-100 tickers from MongoDB
def get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY):
    """
    Connects to MongoDB and retrieves NASDAQ-100 tickers.

    :param mongo_url: MongoDB connection URL
    :return: List of NASDAQ-100 ticker symbols.
    """

    def call_ndaq_100():
        """
        Fetches the list of NASDAQ 100 tickers using the Financial Modeling Prep API and stores it in a MongoDB collection.
        The MongoDB collection is cleared before inserting the updated list of tickers.
        """
        logging.info("Calling NASDAQ 100 to retrieve tickers.")

        def get_jsonparsed_data(url):
            """
            Parses the JSON response from the provided URL.

            :param url: The API endpoint to retrieve data from.
            :return: Parsed JSON data as a dictionary.
            """
            response = urlopen(url)
            data = response.read().decode("utf-8")
            return json.loads(data)

        try:
            # API URL for fetching NASDAQ 100 tickers
            ndaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FINANCIAL_PREP_API_KEY}"  # noqa: E231
            ndaq_stocks = get_jsonparsed_data(ndaq_url)
            logging.info("Successfully retrieved NASDAQ 100 tickers.")
        except Exception as e:
            logging.error(f"Error fetching NASDAQ 100 tickers: {e}")
            return
        try:
            # MongoDB connection details

            db = mongo_client.stock_list
            ndaq100_tickers = db.ndaq100_tickers

            ndaq100_tickers.delete_many({})  # Clear existing data
            ndaq100_tickers.insert_many(ndaq_stocks)  # Insert new data
            logging.info("Successfully inserted NASDAQ 100 tickers into MongoDB.")
        except Exception as e:
            logging.error(f"Error inserting tickers into MongoDB: {e}")

    call_ndaq_100()

    tickers = [
        stock["symbol"] for stock in mongo_client.stock_list.ndaq100_tickers.find()
    ]

    return tickers


# Market status checker helper
def market_status(polygon_client):
    """
    Check market status using the Polygon API.

    :param polygon_client: An instance of the Polygon RESTClient
    :return: Current market status ('open', 'early_hours', 'closed')
    """
    try:
        status = polygon_client.get_market_status()
        if status.exchanges.nasdaq == "open" and status.exchanges.nyse == "open":
            return "open"
        elif status.early_hours:
            return "early_hours"
        else:
            return "closed"
    except Exception as e:
        logging.error(f"Error retrieving market status: {e}")
        return "error"


# Helper to get latest price
def get_latest_price(ticker):
    """
    Fetch the latest price for a given stock ticker using yfinance.

    :param ticker: The stock ticker symbol
    :return: The latest price of the stock
    """
    try:
        ticker_yahoo = yf.Ticker(ticker)
        data = ticker_yahoo.history()

        return round(data["Close"].iloc[-1], 2)
    except Exception as e:
        logging.error(f"Error fetching latest price for {ticker}: {e}")
        return None


def dynamic_period_selector(ticker):
    """
    Determines the best period to use for fetching historical data.

    Args:
    - ticker (str): Stock ticker symbol.

    Returns:
    - str: Optimal period for historical data retrieval.
    """
    periods = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    volatility_scores = []

    for period in periods:
        try:
            data = yf.Ticker(ticker).history(period=period)
            if data.empty:
                continue

            # Calculate metrics for decision-making
            daily_changes = data["Close"].pct_change().dropna()
            volatility = daily_changes.std()
            trend_strength = (
                abs(data["Close"].iloc[-1] - data["Close"].iloc[0])
                / data["Close"].iloc[0]
            )

            # Combine metrics into a single score (weight them as desired)
            score = volatility * 0.7 + trend_strength * 0.3
            volatility_scores.append((period, score))
        except Exception as e:
            print(f"Error fetching data for period {period}: {e}")
            continue

    # Select the period with the highest score

    optimal_period = (
        min(volatility_scores, key=lambda x: x[1])[0] if volatility_scores else "1y"
    )
    return optimal_period


def store_dict_as_json(data_dict, filename, folder_name, logger):
    """
    Stores a dictionary as a JSON file in a specified folder.

    Args:
        data_dict (dict): The dictionary to store.
        filename (str): The name of the JSON file (e.g., "my_data.json").
        folder_name (str, optional): The name of the folder to store the file in.
                                    Defaults to "results_folder".
    """

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logger.info(f"Created folder: {folder_name}")

    # Construct the full filepath
    filepath = os.path.join(folder_name, filename)

    try:
        with open(filepath, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for pretty-printing
        logger.info(f"Dictionary successfully stored as JSON in: {filepath}")
    except Exception as e:
        logger.error(f"Error storing dictionary as JSON: {e}")


def save_df_to_csv(df, filename, folder, logger=None):
    """
    Saves a Pandas DataFrame to a CSV file in the specified folder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the CSV file (e.g., "data.csv").
        folder (str): The path to the folder where the file should be saved.
        logger (logging.Logger, optional): A logger object for logging messages. Defaults to None.

    Raises:
        TypeError: If df is not a Pandas DataFrame.
        ValueError: If filename or folder is empty.
        FileNotFoundError: If the folder does not exist and cannot be created.
        Exception: For other errors during file saving.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame.")
    if not filename:
        raise ValueError("filename cannot be empty.")
    if not folder:
        raise ValueError("folder cannot be empty.")

    # Ensure the folder exists
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            if logger:
                logger.info(f"Created directory: {folder}")
        except OSError as e:
            if logger:
                logger.error(f"Failed to create directory {folder}: {e}")
            raise FileNotFoundError(f"Failed to create directory: {folder}") from e

    # Construct the full file path
    filepath = os.path.join(folder, filename)

    try:
        # Save the DataFrame to CSV
        df.to_csv(filepath, index=True)  # Include index by default
        if logger:
            logger.info(f"DataFrame saved to: {filepath}")
    except Exception as e:
        if logger:
            logger.error(f"Error saving DataFrame to {filepath}: {e}")
        raise Exception(f"Error saving DataFrame to {filepath}") from e


def load_json_to_dict(folder_path, filename):
    """
    Loads a JSON file from a specific folder and returns its content as a Python dictionary.

    Args:
        folder_path (str): The path to the folder containing the JSON file.
        filename (str): The name of the JSON file to load.

    Returns:
        dict: The data loaded from the JSON file as a dictionary.
              Returns None if the file does not exist or an error occurs.
        str : the absolute path to the loaded json file
    """
    filepath = os.path.join(folder_path, filename)
    abs_filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"Loaded {filepath}")
        return data, abs_filepath
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {e}")
        return None, None
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading {filepath}: {e}")
        return None, None


def setup_logging(logs_dir, filename, level=logging.INFO):
    """
    Sets up logging to both a file and the console.

    Args:
        logs_dir (str): The directory where the log file will be stored.
        filename (str): The name of the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
