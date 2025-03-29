import inspect
import pandas as pd
import talib as ta
import numpy as np
import inspect
import logging
import os, sys


# Move the sys.path modification to the top
# Ensure the parent directory is in the path before importing from it
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Configure logging directly in this file
def setup_logging(log_dir, log_file, level=logging.INFO):
    """Sets up logging to a file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging("logs", "store_data.log", level=logging.INFO)


"""
Key Changes and Considerations:

    Vectorization: All calculations (ta.*) and comparisons (>, <) are now done on entire Pandas Series.

    numpy.select: This function is used to efficiently apply the conditional logic ('Buy' if condition A, 'Sell' if condition B, 'Hold' otherwise) across the whole Series.

    Return Value: Functions now modify the input DataFrame data by adding a new signal column (e.g., data['SMA_Signal'] = ...) and return the modified DataFrame. This is more flexible for further analysis.

    No .iloc[-1]: Accessing the last element (.iloc[-1]) is removed, as we operate on all rows.

    ticker Argument: The ticker argument is removed as it's no longer needed within the vectorized function. The function operates solely on the provided data DataFrame.

    Helper Function: A _generate_signals helper simplifies the common np.select pattern.

    Pattern Recognition (CDL)*: A helper _pattern_signals is used for the standard Buy(100)/Sell(-100) logic of TA-Lib patterns. You need to create a vectorized function for each CDL pattern similar to the CDL2CROWS_vectorized example.

    Parameter Defaults: Some TA-Lib functions had unusual defaults in the original code (e.g., SAR, T3, SAREXT). I've used more common defaults in the vectorized versions but kept comments noting the original values. You can easily change these defaults.

    MAMA/MAVP: These functions sometimes require NumPy arrays instead of Series depending on the TA-Lib version. The code includes a try-except block to handle this, converting to .values if necessary and then back to a Pandas Series with the correct index. For MAVP, you need to provide the periods Series/array; the example shows how to create a basic one, but your actual logic for variable periods might be more complex.

    Potentially Flawed Logic: Warnings have been added for indicators where the original Buy/Sell logic seemed unconventional or potentially incorrect (e.g., comparing OBV/AD to 0, fixed levels for ATR/STDDEV/VAR, MINUS_DM/PLUS_DM comparison to 0, HT_TRENDMODE sell condition). Review these based on your actual trading strategy.

    Clarity: Indicator values (like the actual SMA line) can optionally be added to the DataFrame alongside the signals by uncommenting the relevant lines (e.g., # data['SMA'] = sma)."""


# --- Helper Function for Common Logic ---
def _generate_signals(condition_buy, condition_sell, default="Hold"):
    """Uses np.select to generate signals based on boolean conditions."""
    conditions = [condition_buy, condition_sell]
    choices = ["Buy", "Sell"]
    return np.select(conditions, choices, default=default)


# --- Overlap Studies ---


def BBANDS_indicator(data, timeperiod=20):
    """Vectorized Bollinger Bands (BBANDS) indicator signals."""
    upper, middle, lower = ta.BBANDS(data["Close"], timeperiod=timeperiod)
    data["BBANDS_Signal"] = _generate_signals(
        condition_buy=data["Close"] < lower, condition_sell=data["Close"] > upper
    )
    return data["BBANDS_Signal"]


def DEMA_indicator(data, timeperiod=30):
    """Vectorized Double Exponential Moving Average (DEMA) indicator signals."""
    dema = ta.DEMA(data["Close"], timeperiod=timeperiod)
    data["DEMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > dema, condition_sell=data["Close"] < dema
    )
    return data["DEMA_Signal"]


def EMA_indicator(data, timeperiod=30):
    """Vectorized Exponential Moving Average (EMA) indicator signals."""
    ema = ta.EMA(data["Close"], timeperiod=timeperiod)
    data["EMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ema, condition_sell=data["Close"] < ema
    )
    return data["EMA_Signal"]


def HT_TRENDLINE_indicator(data):
    """Vectorized Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) signals."""
    ht_trendline = ta.HT_TRENDLINE(data["Close"])
    data["HT_TRENDLINE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ht_trendline,
        condition_sell=data["Close"] < ht_trendline,
    )
    return data["HT_TRENDLINE_Signal"]


def KAMA_indicator(data, timeperiod=30):
    """Vectorized Kaufman Adaptive Moving Average (KAMA) indicator signals."""
    kama = ta.KAMA(data["Close"], timeperiod=timeperiod)
    data["KAMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > kama, condition_sell=data["Close"] < kama
    )
    return data["KAMA_Signal"]


def MA_indicator(data, timeperiod=30, matype=0):
    """Vectorized Moving average (MA) indicator signals."""
    ma = ta.MA(data["Close"], timeperiod=timeperiod, matype=matype)
    data["MA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ma, condition_sell=data["Close"] < ma
    )
    return data["MA_Signal"]


def MAMA_indicator(data, fastlimit=0.5, slowlimit=0.05):
    """Vectorized MESA Adaptive Moving Average (MAMA) indicator signals."""
    try:
        mama, fama = ta.MAMA(data["Close"], fastlimit=fastlimit, slowlimit=slowlimit)
    except:
        mama_vals, fama_vals = ta.MAMA(
            data["Close"].values, fastlimit=fastlimit, slowlimit=slowlimit
        )
        mama = pd.Series(mama_vals, index=data.index)
        fama = pd.Series(fama_vals, index=data.index)

    data["MAMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > mama, condition_sell=data["Close"] < mama
    )
    return data["MAMA_Signal"]


def MAVP_indicator(data, minperiod=2, maxperiod=30, matype=0):
    if "periods" not in data.columns:
        logger.warning(
            "Warning: 'periods' column not found for MAVP_indicator. Creating a constant period Series (30.0)."
        )
        periods_series = pd.Series(30.0, index=data.index)

        periods_input = periods_series
    else:
        periods_input = data["periods"].astype(float)

    # periods_aligned = (
    #     periods_input.reindex(data.index).fillna(method="ffill").fillna(30.0)
    # )
    periods_aligned = periods_input.reindex(data.index).ffill().fillna(30.0)

    try:
        mavp = ta.MAVP(
            data["Close"],
            periods=periods_aligned,
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=matype,
        )
    except TypeError:
        mavp_vals = ta.MAVP(
            data["Close"].values,
            periods=periods_aligned.values,
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=matype,
        )
        mavp = pd.Series(mavp_vals, index=data.index)

    data["MAVP_Signal"] = _generate_signals(
        condition_buy=data["Close"] > mavp, condition_sell=data["Close"] < mavp
    )
    return data["MAVP_Signal"]


def MIDPOINT_indicator(data, timeperiod=14):
    """Vectorized MidPoint over period (MIDPOINT) indicator signals."""
    midpoint = ta.MIDPOINT(data["Close"], timeperiod=timeperiod)
    data["MIDPOINT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > midpoint, condition_sell=data["Close"] < midpoint
    )
    return data["MIDPOINT_Signal"]


def MIDPRICE_indicator(data, timeperiod=14):
    """Vectorized Midpoint Price over period (MIDPRICE) indicator signals."""
    midprice = ta.MIDPRICE(data["High"], data["Low"], timeperiod=timeperiod)
    data["MIDPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > midprice, condition_sell=data["Close"] < midprice
    )
    return data["MIDPRICE_Signal"]


def SAR_indicator(data, acceleration=0.02, maximum=0.2):
    """Vectorized Parabolic SAR (SAR) indicator signals."""
    sar = ta.SAR(data["High"], data["Low"], acceleration=acceleration, maximum=maximum)
    data["SAR_Signal"] = _generate_signals(
        condition_buy=data["Close"] > sar, condition_sell=data["Close"] < sar
    )
    return data["SAR_Signal"]


def SAREXT_indicator(
    data,
    startvalue=0,
    offsetonreverse=0,
    accelerationinitlong=0.02,
    accelerationlong=0.02,
    accelerationmaxlong=0.2,
    accelerationinitshort=0.02,
    accelerationshort=0.02,
    accelerationmaxshort=0.2,
):
    """Vectorized Parabolic SAR - Extended (SAREXT) indicator signals."""
    sarext = ta.SAREXT(
        data["High"],
        data["Low"],
        startvalue=startvalue,
        offsetonreverse=offsetonreverse,
        accelerationinitlong=accelerationinitlong,
        accelerationlong=accelerationlong,
        accelerationmaxlong=accelerationmaxlong,
        accelerationinitshort=accelerationinitshort,
        accelerationshort=accelerationshort,
        accelerationmaxshort=accelerationmaxshort,
    )
    data["SAREXT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > sarext, condition_sell=data["Close"] < sarext
    )
    return data["SAREXT_Signal"]


def SMA_indicator(data, timeperiod=30):
    """Vectorized Simple Moving Average (SMA) indicator signals."""
    sma = ta.SMA(data["Close"], timeperiod=timeperiod)
    data["SMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > sma, condition_sell=data["Close"] < sma
    )
    return data["SMA_Signal"]


def T3_indicator(data, timeperiod=5, vfactor=0.7):
    """Vectorized Triple Exponential Moving Average (T3) indicator signals."""
    t3 = ta.T3(data["Close"], timeperiod=timeperiod, vfactor=vfactor)
    data["T3_Signal"] = _generate_signals(
        condition_buy=data["Close"] > t3, condition_sell=data["Close"] < t3
    )
    return data["T3_Signal"]


def TEMA_indicator(data, timeperiod=30):
    """Vectorized Triple Exponential Moving Average (TEMA) indicator signals."""
    tema = ta.TEMA(data["Close"], timeperiod=timeperiod)
    data["TEMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > tema, condition_sell=data["Close"] < tema
    )
    return data["TEMA_Signal"]


def TRIMA_indicator(data, timeperiod=30):
    """Vectorized Triangular Moving Average (TRIMA) indicator signals."""
    trima = ta.TRIMA(data["Close"], timeperiod=timeperiod)
    data["TRIMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > trima, condition_sell=data["Close"] < trima
    )
    return data["TRIMA_Signal"]


def WMA_indicator(data, timeperiod=30):
    """Vectorized Weighted Moving Average (WMA) indicator signals."""
    wma = ta.WMA(data["Close"], timeperiod=timeperiod)
    data["WMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > wma, condition_sell=data["Close"] < wma
    )
    return data["WMA_Signal"]


# --- Revised Momentum Indicators ---


def ADX_indicator_v2(data, timeperiod=14, adx_threshold=20):
    """
    Vectorized ADX indicator signals based on DI+/DI- crossover,
    filtered by ADX strength.
    """
    adx = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )

    di_cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    di_cross_down = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))

    is_trending = adx > adx_threshold

    conditions = [
        (di_cross_up) & is_trending,
        (di_cross_down) & is_trending,
    ]
    choices = ["Buy", "Sell"]
    data["ADX_Signal"] = np.select(conditions, choices, default="Hold")

    return data["ADX_Signal"]


def ADXR_indicator_v2(data, timeperiod=14, adx_threshold=20):
    """
    Vectorized ADXR indicator signals. ADXR smooths ADX.
    Using similar DI crossover logic, filtered by ADXR strength.
    Note: Using ADXR > threshold is less common than ADX > threshold for filtering.
    """
    adxr = ta.ADXR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )

    is_trending = adxr > adx_threshold

    conditions = [
        (plus_di > minus_di) & is_trending,
        (minus_di > plus_di) & is_trending,
    ]
    choices = ["Buy", "Sell"]
    data["ADXR_Signal"] = np.select(conditions, choices, default="Hold")

    logger.warning(
        "Warning: Filtering signals based on ADXR > threshold is less common than using ADX."
    )
    return data["ADXR_Signal"]


def CCI_indicator_v2(data, timeperiod=14, buy_level=-100, sell_level=100):
    """
    Vectorized Commodity Channel Index (CCI) indicator signals.
    Standard interpretation: Buy when crossing UP from oversold (< buy_level),
    Sell when crossing DOWN from overbought (> sell_level).
    Simplified version: Buy if < buy_level, Sell if > sell_level.
    """
    cci = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

    data["CCI_Signal"] = _generate_signals(
        condition_buy=cci < buy_level,
        condition_sell=cci > sell_level,
    )
    return data["CCI_Signal"]


def CMO_indicator_v2(data, timeperiod=14, buy_level=-50, sell_level=50):
    """
    Vectorized Chande Momentum Oscillator (CMO) indicator signals.
    Standard interpretation: Buy when oversold (< buy_level),
    Sell when overbought (> sell_level).
    """
    cmo = ta.CMO(data["Close"], timeperiod=timeperiod)

    data["CMO_Signal"] = _generate_signals(
        condition_buy=cmo < buy_level,
        condition_sell=cmo > sell_level,
    )
    return data["CMO_Signal"]


def DX_indicator_v2(data, timeperiod=14, dx_threshold=20):
    """
    Vectorized Directional Movement Index (DX) indicator signals.
    DX measures spread between DI+ and DI-. High DX = Strong trend.
    Using DI+/DI- crossover logic, filtered by DX strength.
    """
    dx = ta.DX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )

    is_trending = dx > dx_threshold

    conditions = [
        (plus_di > minus_di) & is_trending,
        (minus_di > plus_di) & is_trending,
    ]
    choices = ["Buy", "Sell"]
    data["DX_Signal"] = np.select(conditions, choices, default="Hold")

    return data["DX_Signal"]


def PLUS_MINUS_DI_indicator(data, timeperiod=14):
    """
    Vectorized Minus Directional Indicator (MINUS_DI) signals.
    Revised Logic: Sell if DI- is dominant (DI- > DI+).
    """
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )

    data["MINUS_DI_Signal"] = _generate_signals(
        condition_buy=plus_di > minus_di,
        condition_sell=minus_di > plus_di,
    )
    return data["MINUS_DI_Signal"]


# --- Momentum Indicators ---


def ADX_indicator(data, timeperiod=14):
    """Vectorized Average Directional Movement Index (ADX) indicator signals."""
    adx = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["ADX_Signal"] = _generate_signals(
        condition_buy=adx > 25, condition_sell=adx < 20
    )
    return data["ADX_Signal"]


def ADXR_indicator(data, timeperiod=14):
    """Vectorized Average Directional Movement Index Rating (ADXR) indicator signals."""
    adxr = ta.ADXR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["ADXR_Signal"] = _generate_signals(
        condition_buy=adxr > 25, condition_sell=adxr < 20
    )
    return data["ADXR_Signal"]


def APO_indicator(data, fastperiod=12, slowperiod=26, matype=0):
    """Vectorized Absolute Price Oscillator (APO) indicator signals."""
    apo = ta.APO(
        data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype
    )
    data["APO_Signal"] = _generate_signals(
        condition_buy=apo > 0, condition_sell=apo < 0
    )
    return data["APO_Signal"]


def AROON_indicator(data, timeperiod=14):
    """Vectorized Aroon (AROON) indicator signals."""
    aroon_down, aroon_up = ta.AROON(data["High"], data["Low"], timeperiod=timeperiod)
    data["AROON_Signal"] = _generate_signals(
        condition_buy=aroon_up > 70, condition_sell=aroon_down > 70
    )
    return data["AROON_Signal"]


def AROONOSC_indicator(data, timeperiod=14):
    """Vectorized Aroon Oscillator (AROONOSC) indicator signals."""
    aroonosc = ta.AROONOSC(data["High"], data["Low"], timeperiod=timeperiod)
    data["AROONOSC_Signal"] = _generate_signals(
        condition_buy=aroonosc > 0, condition_sell=aroonosc < 0
    )
    return data["AROONOSC_Signal"]


def BOP_indicator(data):
    """Vectorized Balance Of Power (BOP) indicator signals."""
    bop = ta.BOP(data["Open"], data["High"], data["Low"], data["Close"])
    data["BOP_Signal"] = _generate_signals(
        condition_buy=bop > 0, condition_sell=bop < 0
    )
    return data["BOP_Signal"]


def CCI_indicator(data, timeperiod=14):
    """Vectorized Commodity Channel Index (CCI) indicator signals."""
    cci = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["CCI_Signal"] = _generate_signals(
        condition_buy=cci > 100, condition_sell=cci < -100
    )
    return data["CCI_Signal"]


def CMO_indicator(data, timeperiod=14):
    """Vectorized Chande Momentum Oscillator (CMO) indicator signals."""
    cmo = ta.CMO(data["Close"], timeperiod=timeperiod)
    data["CMO_Signal"] = _generate_signals(
        condition_buy=cmo > 50, condition_sell=cmo < -50
    )
    return data["CMO_Signal"]


def DX_indicator(data, timeperiod=14):
    """Vectorized Directional Movement Index (DX) indicator signals."""
    dx = ta.DX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["DX_Signal"] = _generate_signals(condition_buy=dx > 25, condition_sell=dx < 20)
    return data["DX_Signal"]


def MACD_indicator(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """Vectorized Moving Average Convergence/Divergence (MACD) indicator signals."""
    macd, macdsignal, macdhist = ta.MACD(
        data["Close"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    data["MACD_Signal"] = _generate_signals(
        condition_buy=macdhist > 0,
        condition_sell=macdhist < 0,
    )
    return data["MACD_Signal"]


def MACDEXT_indicator(
    data,
    fastperiod=12,
    fastmatype=0,
    slowperiod=26,
    slowmatype=0,
    signalperiod=9,
    signalmatype=0,
):
    """Vectorized MACD with controllable MA type (MACDEXT) indicator signals."""
    macd, macdsignal, macdhist = ta.MACDEXT(
        data["Close"],
        fastperiod=fastperiod,
        fastmatype=fastmatype,
        slowperiod=slowperiod,
        slowmatype=slowmatype,
        signalperiod=signalperiod,
        signalmatype=signalmatype,
    )
    data["MACDEXT_Signal"] = _generate_signals(
        condition_buy=macdhist > 0, condition_sell=macdhist < 0
    )
    return data["MACDEXT_Signal"]


def MACDFIX_indicator(data, signalperiod=9):
    """Vectorized Moving Average Convergence/Divergence Fix 12/26 (MACDFIX) signals."""
    macd, macdsignal, macdhist = ta.MACDFIX(data["Close"], signalperiod=signalperiod)
    data["MACDFIX_Signal"] = _generate_signals(
        condition_buy=macdhist > 0, condition_sell=macdhist < 0
    )
    return data["MACDFIX_Signal"]


def MFI_indicator(data, timeperiod=14):
    """Vectorized Money Flow Index (MFI) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("MFI_indicator requires 'Volume' column in data")

    mfi = ta.MFI(
        data["High"], data["Low"], data["Close"], data["Volume"], timeperiod=timeperiod
    )
    data["MFI_Signal"] = _generate_signals(
        condition_buy=mfi < 20, condition_sell=mfi > 80
    )
    return data["MFI_Signal"]


def MINUS_DI_indicator(data, timeperiod=14):
    """Vectorized Minus Directional Indicator (MINUS_DI) indicator signals."""
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    data["MINUS_DI_Signal"] = _generate_signals(
        condition_buy=minus_di < 20,
        condition_sell=minus_di > 25,
    )
    return data["MINUS_DI_Signal"]


def PLUS_DI_indicator(data, timeperiod=14):
    """Vectorized Plus Directional Indicator (PLUS_DI) indicator signals."""
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    data["PLUS_DI_Signal"] = _generate_signals(
        condition_buy=plus_di > 25,
        condition_sell=plus_di < 20,
    )
    return data["PLUS_DI_Signal"]


def MINUS_DM_indicator(data, timeperiod=14):
    """Vectorized Minus Directional Movement (MINUS_DM) indicator signals."""
    minus_dm = ta.MINUS_DM(data["High"], data["Low"], timeperiod=timeperiod)
    data["MINUS_DM_Signal"] = _generate_signals(
        condition_buy=minus_dm < 0,
        condition_sell=minus_dm > 0,
    )
    logger.warning(
        "Warning: The implemented logic for MINUS_DM_indicator based on the original function seems potentially incorrect."
    )
    return data["MINUS_DM_Signal"]


def MOM_indicator(data, timeperiod=10):
    """Vectorized Momentum (MOM) indicator signals."""
    mom = ta.MOM(data["Close"], timeperiod=timeperiod)
    data["MOM_Signal"] = _generate_signals(
        condition_buy=mom > 0, condition_sell=mom < 0
    )
    return data["MOM_Signal"]


def PLUS_DM_indicator(data, timeperiod=14):
    """Vectorized Plus Directional Movement (PLUS_DM) indicator signals."""
    plus_dm = ta.PLUS_DM(data["High"], data["Low"], timeperiod=timeperiod)
    data["PLUS_DM_Signal"] = _generate_signals(
        condition_buy=plus_dm > 0,
        condition_sell=plus_dm < 0,
    )
    logger.warning(
        "Warning: The implemented logic for PLUS_DM_indicator based on the original function seems potentially incorrect."
    )
    return data["PLUS_DM_Signal"]


def PPO_indicator(data, fastperiod=12, slowperiod=26, matype=0):
    """Vectorized Percentage Price Oscillator (PPO) indicator signals."""
    ppo = ta.PPO(
        data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype
    )
    data["PPO_Signal"] = _generate_signals(
        condition_buy=ppo > 0, condition_sell=ppo < 0
    )
    return data["PPO_Signal"]


def ROC_indicator(data, timeperiod=10):
    """Vectorized Rate of change : ((price/prevPrice)-1)*100 (ROC) signals."""
    roc = ta.ROC(data["Close"], timeperiod=timeperiod)
    data["ROC_Signal"] = _generate_signals(
        condition_buy=roc > 0, condition_sell=roc < 0
    )
    return data["ROC_Signal"]


def ROCP_indicator(data, timeperiod=10):
    """Vectorized Rate of change Percentage: (price-prevPrice)/prevPrice signals."""
    rocp = ta.ROCP(data["Close"], timeperiod=timeperiod)
    data["ROCP_Signal"] = _generate_signals(
        condition_buy=rocp > 0, condition_sell=rocp < 0
    )
    return data["ROCP_Signal"]


def ROCR_indicator(data, timeperiod=10):
    """Vectorized Rate of change ratio: (price/prevPrice) signals."""
    rocr = ta.ROCR(data["Close"], timeperiod=timeperiod)
    data["ROCR_Signal"] = _generate_signals(
        condition_buy=rocr > 1, condition_sell=rocr < 1
    )
    return data["ROCR_Signal"]


def ROCR100_indicator(data, timeperiod=10):
    """Vectorized Rate of change ratio 100 scale: (price/prevPrice)*100 signals."""
    rocr100 = ta.ROCR100(data["Close"], timeperiod=timeperiod)
    data["ROCR100_Signal"] = _generate_signals(
        condition_buy=rocr100 > 100, condition_sell=rocr100 < 100
    )
    return data["ROCR100_Signal"]


def RSI_indicator(data, timeperiod=14):
    """Vectorized Relative Strength Index (RSI) indicator signals."""
    rsi = ta.RSI(data["Close"], timeperiod=timeperiod)
    data["RSI_Signal"] = _generate_signals(
        condition_buy=rsi < 30, condition_sell=rsi > 70
    )
    return data["RSI_Signal"]


def STOCH_indicator(
    data, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
):
    """Vectorized Stochastic (STOCH) indicator signals (using SlowK)."""
    slowk, slowd = ta.STOCH(
        data["High"],
        data["Low"],
        data["Close"],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype,
    )
    data["STOCH_Signal"] = _generate_signals(
        condition_buy=slowk < 20, condition_sell=slowk > 80
    )
    return data["STOCH_Signal"]


def STOCHF_indicator(data, fastk_period=5, fastd_period=3, fastd_matype=0):
    """Vectorized Stochastic Fast (STOCHF) indicator signals (using FastK)."""
    fastk, fastd = ta.STOCHF(
        data["High"],
        data["Low"],
        data["Close"],
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )
    data["STOCHF_Signal"] = _generate_signals(
        condition_buy=fastk < 20, condition_sell=fastk > 80
    )
    return data["STOCHF_Signal"]


def STOCHRSI_indicator(
    data, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
):
    """Vectorized Stochastic Relative Strength Index (STOCHRSI) signals (using FastK)."""
    fastk, fastd = ta.STOCHRSI(
        data["Close"],
        timeperiod=timeperiod,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )
    data["STOCHRSI_Signal"] = _generate_signals(
        condition_buy=fastk < 20, condition_sell=fastk > 80
    )
    return data["STOCHRSI_Signal"]


def TRIX_indicator(data, timeperiod=30):
    """Vectorized 1-day ROC of a Triple Smooth EMA (TRIX) indicator signals."""
    trix = ta.TRIX(data["Close"], timeperiod=timeperiod)
    data["TRIX_Signal"] = _generate_signals(
        condition_buy=trix > 0, condition_sell=trix < 0
    )
    return data["TRIX_Signal"]


def ULTOSC_indicator(data, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """Vectorized Ultimate Oscillator (ULTOSC) indicator signals."""
    ultosc = ta.ULTOSC(
        data["High"],
        data["Low"],
        data["Close"],
        timeperiod1=timeperiod1,
        timeperiod2=timeperiod2,
        timeperiod3=timeperiod3,
    )
    data["ULTOSC_Signal"] = _generate_signals(
        condition_buy=ultosc < 30, condition_sell=ultosc > 70
    )
    return data["ULTOSC_Signal"]


def WILLR_indicator(data, timeperiod=14):
    """Vectorized Williams' %R (WILLR) indicator signals."""
    willr = ta.WILLR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["WILLR_Signal"] = _generate_signals(
        condition_buy=willr < -80, condition_sell=willr > -20
    )
    return data["WILLR_Signal"]


# --- Revised Volume Indicators ---


def AD_indicator_v2(data, ma_period=20):
    """
    Vectorized Chaikin A/D Line (AD) indicator signals.
    Revised logic: Compare AD line to its moving average.
    """
    if "Volume" not in data.columns:
        raise ValueError("AD_indicator requires 'Volume' column in data")
    ad = ta.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    ad_ma = ta.SMA(ad, timeperiod=ma_period)

    data["AD_Signal"] = _generate_signals(
        condition_buy=ad > ad_ma,
        condition_sell=ad < ad_ma,
    )
    logger.warning("Note: AD_indicator revised logic uses AD crossing its SMA.")
    return data["AD_Signal"]


def OBV_indicator_v2(data, ma_period=20):
    """
    Vectorized On Balance Volume (OBV) indicator signals.
    Revised logic: Compare OBV to its moving average.
    """
    if "Volume" not in data.columns:
        raise ValueError("OBV_indicator requires 'Volume' column in data")
    obv = ta.OBV(data["Close"], data["Volume"])
    obv_ma = ta.SMA(obv, timeperiod=ma_period)

    data["OBV_Signal"] = _generate_signals(
        condition_buy=obv > obv_ma,
        condition_sell=obv < obv_ma,
    )
    logger.warning("Note: OBV_indicator revised logic uses OBV crossing its SMA.")
    return data["OBV_Signal"]


# --- Volume Indicators ---


def AD_indicator(data):
    """Vectorized Chaikin A/D Line (AD) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("AD_indicator requires 'Volume' column in data")
    ad = ta.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    data["AD_Signal"] = _generate_signals(condition_buy=ad > 0, condition_sell=ad < 0)
    logger.warning(
        "Warning: The implemented logic for AD_indicator based on the original function might be unconventional."
    )
    return data["AD_Signal"]


def ADOSC_indicator(data, fastperiod=3, slowperiod=10):
    """Vectorized Chaikin A/D Oscillator (ADOSC) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("ADOSC_indicator requires 'Volume' column in data")
    adosc = ta.ADOSC(
        data["High"],
        data["Low"],
        data["Close"],
        data["Volume"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
    )
    data["ADOSC_Signal"] = _generate_signals(
        condition_buy=adosc > 0, condition_sell=adosc < 0
    )
    return data["ADOSC_Signal"]


def OBV_indicator(data):
    """Vectorized On Balance Volume (OBV) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("OBV_indicator requires 'Volume' column in data")
    obv = ta.OBV(data["Close"], data["Volume"])
    data["OBV_Signal"] = _generate_signals(
        condition_buy=obv > 0, condition_sell=obv < 0
    )
    logger.warning(
        "Warning: The implemented logic for OBV_indicator based on the original function might be unconventional."
    )
    return data["OBV_Signal"]


# --- Revised Cycle Indicators ---


def HT_TRENDMODE_indicator_v2(data):
    """
    Vectorized Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) signals.
    Revised logic: Buy in Trend Mode (1), Sell in Cycle Mode (0).
    """
    ht_trendmode = ta.HT_TRENDMODE(data["Close"])

    data["HT_TRENDMODE_Signal"] = _generate_signals(
        condition_buy=ht_trendmode == 1,
        condition_sell=ht_trendmode == 0,
    )
    logger.warning(
        "Note: HT_TRENDMODE_indicator revised logic: Buy on Trend(1), Sell on Cycle(0)."
    )
    return data["HT_TRENDMODE_Signal"]


# --- Cycle Indicators ---


def HT_DCPERIOD_indicator(data):
    """Vectorized Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD) signals."""
    ht_dcperiod = ta.HT_DCPERIOD(data["Close"])
    data["HT_DCPERIOD_Signal"] = _generate_signals(
        condition_buy=ht_dcperiod > 20,
        condition_sell=ht_dcperiod < 10,
    )
    return data["HT_DCPERIOD_Signal"]


def HT_DCPHASE_indicator(data):
    """Vectorized Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) signals."""
    ht_dcphase = ta.HT_DCPHASE(data["Close"])
    data["HT_DCPHASE_Signal"] = _generate_signals(
        condition_buy=ht_dcphase > 0, condition_sell=ht_dcphase < 0
    )
    return data["HT_DCPHASE_Signal"]


def HT_PHASOR_indicator(data):
    """Vectorized Hilbert Transform - Phasor Components (HT_PHASOR) signals (using inphase)."""
    inphase, quadrature = ta.HT_PHASOR(data["Close"])
    data["HT_PHASOR_Signal"] = _generate_signals(
        condition_buy=inphase > 0, condition_sell=inphase < 0
    )
    return data["HT_PHASOR_Signal"]


def HT_SINE_indicator(data):
    """Vectorized Hilbert Transform - SineWave (HT_SINE) indicator signals (using sine)."""
    sine, leadsine = ta.HT_SINE(data["Close"])
    data["HT_SINE_Signal"] = _generate_signals(
        condition_buy=sine > 0,
        condition_sell=sine < 0,
    )
    return data["HT_SINE_Signal"]


def HT_TRENDMODE_indicator(data):
    """Vectorized Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) signals."""
    ht_trendmode = ta.HT_TRENDMODE(data["Close"])
    data["HT_TRENDMODE_Signal"] = _generate_signals(
        condition_buy=ht_trendmode > 0,
        condition_sell=ht_trendmode < 0,
    )
    logger.warning(
        "Warning: Original HT_TRENDMODE_indicator logic might be flawed (Sell condition never met)."
    )
    return data["HT_TRENDMODE_Signal"]


# --- Price Transform ---


def AVGPRICE_indicator(data):
    """Vectorized Average Price (AVGPRICE) indicator signals."""
    avgprice = ta.AVGPRICE(data["Open"], data["High"], data["Low"], data["Close"])
    data["AVGPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > avgprice, condition_sell=data["Close"] < avgprice
    )
    return data["AVGPRICE_Signal"]


def MEDPRICE_indicator(data):
    """Vectorized Median Price (MEDPRICE) indicator signals."""
    medprice = ta.MEDPRICE(data["High"], data["Low"])
    data["MEDPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > medprice, condition_sell=data["Close"] < medprice
    )
    return data["MEDPRICE_Signal"]


def TYPPRICE_indicator(data):
    """Vectorized Typical Price (TYPPRICE) indicator signals."""
    typprice = ta.TYPPRICE(data["High"], data["Low"], data["Close"])
    data["TYPPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > typprice, condition_sell=data["Close"] < typprice
    )
    return data["TYPPRICE_Signal"]


def WCLPRICE_indicator(data):
    """Vectorized Weighted Close Price (WCLPRICE) indicator signals."""
    wclprice = ta.WCLPRICE(data["High"], data["Low"], data["Close"])
    data["WCLPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > wclprice, condition_sell=data["Close"] < wclprice
    )
    return data["WCLPRICE_Signal"]


# --- Revised Volatility Indicators ---


def ATR_indicator_v2(data, timeperiod=14, ma_period=14):
    """
    Vectorized Average True Range (ATR) indicator signals.
    Revised logic: Compare ATR to its moving average.
    WARNING: This is not a standard signal generation technique for ATR.
    """
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    atr_ma = ta.SMA(atr, timeperiod=ma_period)

    data["ATR_Signal"] = _generate_signals(
        condition_buy=atr > atr_ma,
        condition_sell=atr < atr_ma,
    )
    logger.warning(
        "Warning: ATR_indicator revised logic (ATR vs MA) is unconventional for Buy/Sell signals."
    )
    return data["ATR_Signal"]


def NATR_indicator_v2(data, timeperiod=14, ma_period=14):
    """
    Vectorized Normalized Average True Range (NATR) indicator signals.
    Revised logic: Compare NATR to its moving average.
    WARNING: This is not a standard signal generation technique for NATR.
    """
    natr = ta.NATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    natr_ma = ta.SMA(natr, timeperiod=ma_period)

    data["NATR_Signal"] = _generate_signals(
        condition_buy=natr > natr_ma,
        condition_sell=natr < natr_ma,
    )
    logger.warning(
        "Warning: NATR_indicator revised logic (NATR vs MA) is unconventional for Buy/Sell signals."
    )
    return data["NATR_Signal"]


def TRANGE_indicator_v2(data, ma_period=14):
    """
    Vectorized True Range (TRANGE) indicator signals.
    Revised logic: Compare TRANGE to its moving average.
    WARNING: This is not a standard signal generation technique for TRANGE.
    """
    trange = ta.TRANGE(data["High"], data["Low"], data["Close"])
    trange_ma = ta.SMA(trange, timeperiod=ma_period)

    data["TRANGE_Signal"] = _generate_signals(
        condition_buy=trange > trange_ma,
        condition_sell=trange < trange_ma,
    )
    logger.warning(
        "Warning: TRANGE_indicator revised logic (TRANGE vs MA) is unconventional for Buy/Sell signals."
    )
    return data["TRANGE_Signal"]


# --- Volatility Indicators ---


def ATR_indicator(data, timeperiod=14):
    """Vectorized Average True Range (ATR) indicator signals."""
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["ATR_Signal"] = _generate_signals(
        condition_buy=atr > 20, condition_sell=atr < 10
    )
    logger.warning(
        "Warning: Using fixed ATR levels (10, 20) for Buy/Sell signals in ATR_indicator is unconventional."
    )
    return data["ATR_Signal"]


def NATR_indicator(data, timeperiod=14):
    """Vectorized Normalized Average True Range (NATR) indicator signals."""
    natr = ta.NATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    data["NATR_Signal"] = _generate_signals(
        condition_buy=natr > 20, condition_sell=natr < 10
    )
    logger.warning(
        "Warning: Using fixed NATR levels (10, 20) for Buy/Sell signals in NATR_indicator is unconventional."
    )
    return data["NATR_Signal"]


def TRANGE_indicator(data):
    """Vectorized True Range (TRANGE) indicator signals."""
    trange = ta.TRANGE(data["High"], data["Low"], data["Close"])
    data["TRANGE_Signal"] = _generate_signals(
        condition_buy=trange > 20, condition_sell=trange < 10
    )
    logger.warning(
        "Warning: Using fixed TRANGE levels (10, 20) for Buy/Sell signals in TRANGE_indicator is unconventional."
    )
    return data["TRANGE_Signal"]


# --- Pattern Recognition ---


def _pattern_signals(pattern_series):
    """Helper for standard pattern recognition signals."""
    return _generate_signals(
        condition_buy=pattern_series > 0,
        condition_sell=pattern_series < 0,
    )


def CDL2CROWS_indicator(data):
    """Vectorized Two Crows (CDL2CROWS) indicator signals."""
    pattern = ta.CDL2CROWS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL2CROWS_Signal"] = _pattern_signals(pattern)
    return data["CDL2CROWS_Signal"]


def CDL3BLACKCROWS_indicator(data):
    """Vectorized Three Black Crows (CDL3BLACKCROWS) indicator signals."""
    pattern = ta.CDL3BLACKCROWS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3BLACKCROWS_Signal"] = _pattern_signals(pattern)
    return data["CDL3BLACKCROWS_Signal"]


def CDL3INSIDE_indicator(data):
    """Vectorized Three Inside Up/Down (CDL3INSIDE) indicator signals."""
    pattern = ta.CDL3INSIDE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3INSIDE_Signal"] = _pattern_signals(pattern)
    return data["CDL3INSIDE_Signal"]


def CDL3LINESTRIKE_indicator(data):
    """Vectorized Three-Line Strike (CDL3LINESTRIKE) indicator signals."""
    pattern = ta.CDL3LINESTRIKE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3LINESTRIKE_Signal"] = _pattern_signals(pattern)
    return data["CDL3LINESTRIKE_Signal"]


def CDL3OUTSIDE_indicator(data):
    """Vectorized Three Outside Up/Down (CDL3OUTSIDE) indicator signals."""
    pattern = ta.CDL3OUTSIDE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3OUTSIDE_Signal"] = _pattern_signals(pattern)
    return data["CDL3OUTSIDE_Signal"]


def CDL3STARSINSOUTH_indicator(data):
    """Vectorized Three Stars In The South (CDL3STARSINSOUTH) indicator signals."""
    pattern = ta.CDL3STARSINSOUTH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDL3STARSINSOUTH_Signal"] = _pattern_signals(pattern)
    return data["CDL3STARSINSOUTH_Signal"]


def CDL3WHITESOLDIERS_indicator(data):
    """Vectorized Three Advancing White Soldiers (CDL3WHITESOLDIERS) indicator signals."""
    pattern = ta.CDL3WHITESOLDIERS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDL3WHITESOLDIERS_Signal"] = _pattern_signals(pattern)
    return data["CDL3WHITESOLDIERS_Signal"]


def CDLABANDONEDBABY_indicator(data, penetration=0):
    """Vectorized Abandoned Baby (CDLABANDONEDBABY) indicator signals."""
    pattern = ta.CDLABANDONEDBABY(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLABANDONEDBABY_Signal"] = _pattern_signals(pattern)
    return data["CDLABANDONEDBABY_Signal"]


def CDLADVANCEBLOCK_indicator(data):
    """Vectorized Advance Block (CDLADVANCEBLOCK) indicator signals."""
    pattern = ta.CDLADVANCEBLOCK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLADVANCEBLOCK_Signal"] = _pattern_signals(pattern)
    return data["CDLADVANCEBLOCK_Signal"]


def CDLBELTHOLD_indicator(data):
    """Vectorized Belt-hold (CDLBELTHOLD) indicator signals."""
    pattern = ta.CDLBELTHOLD(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLBELTHOLD_Signal"] = _pattern_signals(pattern)
    return data["CDLBELTHOLD_Signal"]


def CDLBREAKAWAY_indicator(data):
    """Vectorized Breakaway (CDLBREAKAWAY) indicator signals."""
    pattern = ta.CDLBREAKAWAY(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLBREAKAWAY_Signal"] = _pattern_signals(pattern)
    return data["CDLBREAKAWAY_Signal"]


def CDLCLOSINGMARUBOZU_indicator(data):
    """Vectorized Closing Marubozu (CDLCLOSINGMARUBOZU) indicator signals."""
    pattern = ta.CDLCLOSINGMARUBOZU(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCLOSINGMARUBOZU_Signal"] = _pattern_signals(pattern)
    return data["CDLCLOSINGMARUBOZU_Signal"]


def CDLCONCEALBABYSWALL_indicator(data):
    """Vectorized Concealing Baby Swallow (CDLCONCEALBABYSWALL) indicator signals."""
    pattern = ta.CDLCONCEALBABYSWALL(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCONCEALBABYSWALL_Signal"] = _pattern_signals(pattern)
    return data["CDLCONCEALBABYSWALL_Signal"]


def CDLCOUNTERATTACK_indicator(data):
    """Vectorized Counterattack (CDLCOUNTERATTACK) indicator signals."""
    pattern = ta.CDLCOUNTERATTACK(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCOUNTERATTACK_Signal"] = _pattern_signals(pattern)
    return data["CDLCOUNTERATTACK_Signal"]


def CDLDARKCLOUDCOVER_indicator(data, penetration=0):
    """Vectorized Dark Cloud Cover (CDLDARKCLOUDCOVER) indicator signals."""
    pattern = ta.CDLDARKCLOUDCOVER(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLDARKCLOUDCOVER_Signal"] = _pattern_signals(pattern)
    return data["CDLDARKCLOUDCOVER_Signal"]


def CDLDOJI_indicator(data):
    """Vectorized Doji (CDLDOJI) indicator signals."""
    pattern = ta.CDLDOJI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLDOJI_Signal"] = _pattern_signals(pattern)
    return data["CDLDOJI_Signal"]


def CDLDOJISTAR_indicator(data):
    """Vectorized Doji Star (CDLDOJISTAR) indicator signals."""
    pattern = ta.CDLDOJISTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLDOJISTAR_Signal"]


def CDLDRAGONFLYDOJI_indicator(data):
    """Vectorized Dragonfly Doji (CDLDRAGONFLYDOJI) indicator signals."""
    pattern = ta.CDLDRAGONFLYDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLDRAGONFLYDOJI_Signal"] = _pattern_signals(pattern)
    return data["CDLDRAGONFLYDOJI_Signal"]


def CDLENGULFING_indicator(data):
    """Vectorized Engulfing Pattern (CDLENGULFING) indicator signals."""
    pattern = ta.CDLENGULFING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLENGULFING_Signal"] = _pattern_signals(pattern)
    return data["CDLENGULFING_Signal"]


def CDLEVENINGDOJISTAR_indicator(data, penetration=0):
    """Vectorized Evening Doji Star (CDLEVENINGDOJISTAR) indicator signals."""
    pattern = ta.CDLEVENINGDOJISTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLEVENINGDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLEVENINGDOJISTAR_Signal"]


def CDLEVENINGSTAR_indicator(data, penetration=0):
    """Vectorized Evening Star (CDLEVENINGSTAR) indicator signals."""
    pattern = ta.CDLEVENINGSTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLEVENINGSTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLEVENINGSTAR_Signal"]


def CDLGAPSIDESIDEWHITE_indicator(data):
    """Vectorized Up/Down-gap side-by-side white lines (CDLGAPSIDESIDEWHITE) indicator signals."""
    pattern = ta.CDLGAPSIDESIDEWHITE(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLGAPSIDESIDEWHITE_Signal"] = _pattern_signals(pattern)
    return data["CDLGAPSIDESIDEWHITE_Signal"]


def CDLGRAVESTONEDOJI_indicator(data):
    """Vectorized Gravestone Doji (CDLGRAVESTONEDOJI) indicator signals."""
    pattern = ta.CDLGRAVESTONEDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLGRAVESTONEDOJI_Signal"] = _pattern_signals(pattern)
    return data["CDLGRAVESTONEDOJI_Signal"]


def CDLHAMMER_indicator(data):
    """Vectorized Hammer (CDLHAMMER) indicator signals."""
    pattern = ta.CDLHAMMER(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHAMMER_Signal"] = _pattern_signals(pattern)
    return data["CDLHAMMER_Signal"]


def CDLHANGINGMAN_indicator(data):
    """Vectorized Hanging Man (CDLHANGINGMAN) indicator signals."""
    pattern = ta.CDLHANGINGMAN(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHANGINGMAN_Signal"] = _pattern_signals(pattern)
    return data["CDLHANGINGMAN_Signal"]


def CDLHARAMI_indicator(data):
    """Vectorized Harami Pattern (CDLHARAMI) indicator signals."""
    pattern = ta.CDLHARAMI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHARAMI_Signal"] = _pattern_signals(pattern)
    return data["CDLHARAMI_Signal"]


def CDLHARAMICROSS_indicator(data):
    """Vectorized Harami Cross Pattern (CDLHARAMICROSS) indicator signals."""
    pattern = ta.CDLHARAMICROSS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHARAMICROSS_Signal"] = _pattern_signals(pattern)
    return data["CDLHARAMICROSS_Signal"]


def CDLHIGHWAVE_indicator(data):
    """Vectorized High-Wave Candle (CDLHIGHWAVE) indicator signals."""
    pattern = ta.CDLHIGHWAVE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIGHWAVE_Signal"] = _pattern_signals(pattern)
    return data["CDLHIGHWAVE_Signal"]


def CDLHIKKAKE_indicator(data):
    """Vectorized Hikkake Pattern (CDLHIKKAKE) indicator signals."""
    pattern = ta.CDLHIKKAKE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIKKAKE_Signal"] = _pattern_signals(pattern)
    return data["CDLHIKKAKE_Signal"]


def CDLHIKKAKEMOD_indicator(data):
    """Vectorized Modified Hikkake Pattern (CDLHIKKAKEMOD) indicator signals."""
    pattern = ta.CDLHIKKAKEMOD(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIKKAKEMOD_Signal"] = _pattern_signals(pattern)
    return data["CDLHIKKAKEMOD_Signal"]


def CDLHOMINGPIGEON_indicator(data):
    """Vectorized Homing Pigeon (CDLHOMINGPIGEON) indicator signals."""
    pattern = ta.CDLHOMINGPIGEON(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHOMINGPIGEON_Signal"] = _pattern_signals(pattern)
    return data["CDLHOMINGPIGEON_Signal"]


def CDLIDENTICAL3CROWS_indicator(data):
    """Vectorized Identical Three Crows (CDLIDENTICAL3CROWS) indicator signals."""
    pattern = ta.CDLIDENTICAL3CROWS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLIDENTICAL3CROWS_Signal"] = _pattern_signals(pattern)
    return data["CDLIDENTICAL3CROWS_Signal"]


def CDLINNECK_indicator(data):
    """Vectorized In-Neck Pattern (CDLINNECK) indicator signals."""
    pattern = ta.CDLINNECK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLINNECK_Signal"] = _pattern_signals(pattern)
    return data["CDLINNECK_Signal"]


def CDLINVERTEDHAMMER_indicator(data):
    """Vectorized Inverted Hammer (CDLINVERTEDHAMMER) indicator signals."""
    pattern = ta.CDLINVERTEDHAMMER(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLINVERTEDHAMMER_Signal"] = _pattern_signals(pattern)
    return data["CDLINVERTEDHAMMER_Signal"]


def CDLKICKING_indicator(data):
    """Vectorized Kicking (CDLKICKING) indicator signals."""
    pattern = ta.CDLKICKING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLKICKING_Signal"] = _pattern_signals(pattern)
    return data["CDLKICKING_Signal"]


def CDLKICKINGBYLENGTH_indicator(data):
    """Vectorized Kicking - bull/bear determined by the longer marubozu (CDLKICKINGBYLENGTH) indicator signals."""
    pattern = ta.CDLKICKINGBYLENGTH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLKICKINGBYLENGTH_Signal"] = _pattern_signals(pattern)
    return data["CDLKICKINGBYLENGTH_Signal"]


def CDLLADDERBOTTOM_indicator(data):
    """Vectorized Ladder Bottom (CDLLADDERBOTTOM) indicator signals."""
    pattern = ta.CDLLADDERBOTTOM(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLLADDERBOTTOM_Signal"] = _pattern_signals(pattern)
    return data["CDLLADDERBOTTOM_Signal"]


def CDLLONGLEGGEDDOJI_indicator(data):
    """Vectorized Long Legged Doji (CDLLONGLEGGEDDOJI) indicator signals."""
    pattern = ta.CDLLONGLEGGEDDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLLONGLEGGEDDOJI_Signal"] = _pattern_signals(pattern)
    return data["CDLLONGLEGGEDDOJI_Signal"]


def CDLLONGLINE_indicator(data):
    """Vectorized Long Line Candle (CDLLONGLINE) indicator signals."""
    pattern = ta.CDLLONGLINE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLLONGLINE_Signal"] = _pattern_signals(pattern)
    return data["CDLLONGLINE_Signal"]


def CDLMARUBOZU_indicator(data):
    """Vectorized Marubozu (CDLMARUBOZU) indicator signals."""
    pattern = ta.CDLMARUBOZU(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLMARUBOZU_Signal"] = _pattern_signals(pattern)
    return data["CDLMARUBOZU_Signal"]


def CDLMATCHINGLOW_indicator(data):
    """Vectorized Matching Low (CDLMATCHINGLOW) indicator signals."""
    pattern = ta.CDLMATCHINGLOW(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLMATCHINGLOW_Signal"] = _pattern_signals(pattern)
    return data["CDLMATCHINGLOW_Signal"]


def CDLMATHOLD_indicator(data, penetration=0):
    """Vectorized Mat Hold (CDLMATHOLD) indicator signals."""
    pattern = ta.CDLMATHOLD(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMATHOLD_Signal"] = _pattern_signals(pattern)
    return data["CDLMATHOLD_Signal"]


def CDLMORNINGDOJISTAR_indicator(data, penetration=0):
    """Vectorized Morning Doji Star (CDLMORNINGDOJISTAR) indicator signals."""
    pattern = ta.CDLMORNINGDOJISTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMORNINGDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLMORNINGDOJISTAR_Signal"]


def CDLMORNINGSTAR_indicator(data, penetration=0):
    """Vectorized Morning Star (CDLMORNINGSTAR) indicator signals."""
    pattern = ta.CDLMORNINGSTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMORNINGSTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLMORNINGSTAR_Signal"]


def CDLONNECK_indicator(data):
    """Vectorized On-Neck Pattern (CDLONNECK) indicator signals."""
    pattern = ta.CDLONNECK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLONNECK_Signal"] = _pattern_signals(pattern)
    return data["CDLONNECK_Signal"]


def CDLPIERCING_indicator(data):
    """Vectorized Piercing Pattern (CDLPIERCING) indicator signals."""
    pattern = ta.CDLPIERCING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLPIERCING_Signal"] = _pattern_signals(pattern)
    return data["CDLPIERCING_Signal"]


def CDLRICKSHAWMAN_indicator(data):
    """Vectorized Rickshaw Man (CDLRICKSHAWMAN) indicator signals."""
    pattern = ta.CDLRICKSHAWMAN(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLRICKSHAWMAN_Signal"] = _pattern_signals(pattern)
    return data["CDLRICKSHAWMAN_Signal"]


def CDLRISEFALL3METHODS_indicator(data):
    """Vectorized Rising/Falling Three Methods (CDLRISEFALL3METHODS) indicator signals."""
    pattern = ta.CDLRISEFALL3METHODS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLRISEFALL3METHODS_Signal"] = _pattern_signals(pattern)
    return data["CDLRISEFALL3METHODS_Signal"]


def CDLSEPARATINGLINES_indicator(data):
    """Vectorized Separating Lines (CDLSEPARATINGLINES) indicator signals."""
    pattern = ta.CDLSEPARATINGLINES(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSEPARATINGLINES_Signal"] = _pattern_signals(pattern)
    return data["CDLSEPARATINGLINES_Signal"]


def CDLSHOOTINGSTAR_indicator(data):
    """Vectorized Shooting Star (CDLSHOOTINGSTAR) indicator signals."""
    pattern = ta.CDLSHOOTINGSTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSHOOTINGSTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLSHOOTINGSTAR_Signal"]


def CDLSHORTLINE_indicator(data):
    """Vectorized Short Line Candle (CDLSHORTLINE) indicator signals."""
    pattern = ta.CDLSHORTLINE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSHORTLINE_Signal"] = _pattern_signals(pattern)
    return data["CDLSHORTLINE_Signal"]


def CDLSPINNINGTOP_indicator(data):
    """Vectorized Spinning Top (CDLSPINNINGTOP) indicator signals."""
    pattern = ta.CDLSPINNINGTOP(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSPINNINGTOP_Signal"] = _pattern_signals(pattern)
    return data["CDLSPINNINGTOP_Signal"]


def CDLSTALLEDPATTERN_indicator(data):
    """Vectorized Stalled Pattern (CDLSTALLEDPATTERN) indicator signals."""
    pattern = ta.CDLSTALLEDPATTERN(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSTALLEDPATTERN_Signal"] = _pattern_signals(pattern)
    return data["CDLSTALLEDPATTERN_Signal"]


def CDLSTICKSANDWICH_indicator(data):
    """Vectorized Stick Sandwich (CDLSTICKSANDWICH) indicator signals."""
    pattern = ta.CDLSTICKSANDWICH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSTICKSANDWICH_Signal"] = _pattern_signals(pattern)
    return data["CDLSTICKSANDWICH_Signal"]


def CDLTAKURI_indicator(data):
    """Vectorized Takuri (Dragonfly Doji with very long lower shadow) (CDLTAKURI) indicator signals."""
    pattern = ta.CDLTAKURI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTAKURI_Signal"] = _pattern_signals(pattern)
    return data["CDLTAKURI_Signal"]


def CDLTASUKIGAP_indicator(data):
    """Vectorized Tasuki Gap (CDLTASUKIGAP) indicator signals."""
    pattern = ta.CDLTASUKIGAP(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTASUKIGAP_Signal"] = _pattern_signals(pattern)
    return data["CDLTASUKIGAP_Signal"]


def CDLTHRUSTING_indicator(data):
    """Vectorized Thrusting Pattern (CDLTHRUSTING) indicator signals."""
    pattern = ta.CDLTHRUSTING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTHRUSTING_Signal"] = _pattern_signals(pattern)
    return data["CDLTHRUSTING_Signal"]


def CDLTRISTAR_indicator(data):
    """Vectorized Tristar Pattern (CDLTRISTAR) indicator signals."""
    pattern = ta.CDLTRISTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTRISTAR_Signal"] = _pattern_signals(pattern)
    return data["CDLTRISTAR_Signal"]


def CDLUNIQUE3RIVER_indicator(data):
    """Vectorized Unique 3 River (CDLUNIQUE3RIVER) indicator signals."""
    pattern = ta.CDLUNIQUE3RIVER(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLUNIQUE3RIVER_Signal"] = _pattern_signals(pattern)
    return data["CDLUNIQUE3RIVER_Signal"]


def CDLUPSIDEGAP2CROWS_indicator(data):
    """Vectorized Upside Gap Two Crows (CDLUPSIDEGAP2CROWS) indicator signals."""
    pattern = ta.CDLUPSIDEGAP2CROWS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLUPSIDEGAP2CROWS_Signal"] = _pattern_signals(pattern)
    return data["CDLUPSIDEGAP2CROWS_Signal"]


def CDLXSIDEGAP3METHODS_indicator(data):
    """Vectorized Upside/Downside Gap Three Methods (CDLXSIDEGAP3METHODS) indicator signals."""
    pattern = ta.CDLXSIDEGAP3METHODS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLXSIDEGAP3METHODS_Signal"] = _pattern_signals(pattern)
    return data["CDLXSIDEGAP3METHODS_Signal"]


# --- Revised Statistic Functions ---


def BETA_indicator_v2(data, timeperiod=5):
    """
    Vectorized Beta (BETA) indicator signals.
    Logic: Beta > 1 implies higher volatility than benchmark (Low price here).
    Signal interpretation (Buy/Sell) is highly strategy-dependent.
    Keeping original logic but adding stronger warning.
    """
    beta = ta.BETA(data["High"], data["Low"], timeperiod=timeperiod)
    data["BETA_Signal"] = _generate_signals(
        condition_buy=beta > 1, condition_sell=beta < 1
    )
    logger.warning(
        "Warning: BETA_indicator Buy/Sell signals based on Beta > 1 or < 1 are highly context-dependent and may not be meaningful."
    )
    return data["BETA_Signal"]


def CORREL_indicator_v2(data, timeperiod=30):
    """
    Vectorized Pearson's Correlation Coefficient (CORREL) indicator signals.
    Logic: Measures correlation between High and Low.
    Signal interpretation (Buy/Sell based on >0.5 or <-0.5) is arbitrary.
    Keeping original logic but adding stronger warning.
    """
    correl = ta.CORREL(data["High"], data["Low"], timeperiod=timeperiod)
    data["CORREL_Signal"] = _generate_signals(
        condition_buy=correl > 0.5, condition_sell=correl < -0.5
    )
    logger.warning(
        "Warning: CORREL_indicator Buy/Sell signals based on fixed correlation levels are arbitrary and context-dependent."
    )
    return data["CORREL_Signal"]


def LINEARREG_INTERCEPT_indicator_v2(data, timeperiod=14):
    """
    Vectorized Linear Regression Intercept (LINEARREG_INTERCEPT) indicator signals.
    Revised logic: Compares Close to the forecast value (LINEARREG), not the intercept.
    This makes it logically identical to LINEARREG_indicator.
    """
    linearreg = ta.LINEARREG(data["Close"], timeperiod=timeperiod)

    data["LINEARREG_INTERCEPT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg,
        condition_sell=data["Close"] < linearreg,
    )
    logger.warning(
        "Note: LINEARREG_INTERCEPT_indicator logic revised to compare Close vs LINEARREG (forecast), making it equivalent to LINEARREG_indicator."
    )
    return data["LINEARREG_INTERCEPT_Signal"]


def STDDEV_indicator_v2(data, timeperiod=20, nbdev=1, ma_period=20):
    """
    Vectorized Standard Deviation (STDDEV) indicator signals.
    Revised logic: Compare STDDEV to its moving average.
    WARNING: This is not a standard signal generation technique for STDDEV.
    """
    stddev = ta.STDDEV(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    stddev_ma = ta.SMA(stddev, timeperiod=ma_period)

    data["STDDEV_Signal"] = _generate_signals(
        condition_buy=stddev > stddev_ma,
        condition_sell=stddev < stddev_ma,
    )
    logger.warning(
        "Warning: STDDEV_indicator revised logic (STDDEV vs MA) is unconventional for Buy/Sell signals."
    )
    return data["STDDEV_Signal"]


def VAR_indicator_v2(data, timeperiod=5, nbdev=1, ma_period=5):
    """
    Vectorized Variance (VAR) indicator signals.
    Revised logic: Compare VAR to its moving average.
    WARNING: This is not a standard signal generation technique for VAR.
    """
    var = ta.VAR(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    var_ma = ta.SMA(var, timeperiod=ma_period)

    data["VAR_Signal"] = _generate_signals(
        condition_buy=var > var_ma,
        condition_sell=var < var_ma,
    )
    logger.warning(
        "Warning: VAR_indicator revised logic (VAR vs MA) is unconventional for Buy/Sell signals."
    )
    return data["VAR_Signal"]


# --- Statistic Functions ---


def BETA_indicator(data, timeperiod=5):
    """Vectorized Beta (BETA) indicator signals."""
    beta = ta.BETA(data["High"], data["Low"], timeperiod=timeperiod)
    data["BETA_Signal"] = _generate_signals(
        condition_buy=beta > 1, condition_sell=beta < 1
    )
    return data["BETA_Signal"]


def CORREL_indicator(data, timeperiod=30):
    """Vectorized Pearson's Correlation Coefficient (CORREL) indicator signals."""
    correl = ta.CORREL(data["High"], data["Low"], timeperiod=timeperiod)
    data["CORREL_Signal"] = _generate_signals(
        condition_buy=correl > 0.5, condition_sell=correl < -0.5
    )
    return data["CORREL_Signal"]


def LINEARREG_indicator(data, timeperiod=14):
    """Vectorized Linear Regression (LINEARREG) indicator signals."""
    linearreg = ta.LINEARREG(data["Close"], timeperiod=timeperiod)
    data["LINEARREG_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg,
        condition_sell=data["Close"] < linearreg,
    )
    return data["LINEARREG_Signal"]


def LINEARREG_ANGLE_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Angle (LINEARREG_ANGLE) indicator signals."""
    linearreg_angle = ta.LINEARREG_ANGLE(data["Close"], timeperiod=timeperiod)
    data["LINEARREG_ANGLE_Signal"] = _generate_signals(
        condition_buy=linearreg_angle > 0,
        condition_sell=linearreg_angle < 0,
    )
    return data["LINEARREG_ANGLE_Signal"]


def LINEARREG_INTERCEPT_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Intercept (LINEARREG_INTERCEPT) indicator signals."""
    linearreg_intercept = ta.LINEARREG_INTERCEPT(data["Close"], timeperiod=timeperiod)
    data["LINEARREG_INTERCEPT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg_intercept,
        condition_sell=data["Close"] < linearreg_intercept,
    )
    logger.warning(
        "Warning: Comparing Close to LINEARREG_INTERCEPT for signals in LINEARREG_INTERCEPT_indicator might be unconventional."
    )
    return data["LINEARREG_INTERCEPT_Signal"]


def LINEARREG_SLOPE_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Slope (LINEARREG_SLOPE) indicator signals."""
    linearreg_slope = ta.LINEARREG_SLOPE(data["Close"], timeperiod=timeperiod)
    data["LINEARREG_SLOPE_Signal"] = _generate_signals(
        condition_buy=linearreg_slope > 0,
        condition_sell=linearreg_slope < 0,
    )
    return data["LINEARREG_SLOPE_Signal"]


def STDDEV_indicator(data, timeperiod=20, nbdev=1):
    """Vectorized Standard Deviation (STDDEV) indicator signals."""
    stddev = ta.STDDEV(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    data["STDDEV_Signal"] = _generate_signals(
        condition_buy=stddev > 20, condition_sell=stddev < 10
    )
    logger.warning(
        "Warning: Using fixed STDDEV levels (10, 20) for Buy/Sell signals in STDDEV_indicator is unconventional."
    )
    return data["STDDEV_Signal"]


def TSF_indicator(data, timeperiod=14):
    """Vectorized Time Series Forecast (TSF) indicator signals."""
    tsf = ta.TSF(data["Close"], timeperiod=timeperiod)
    data["TSF_Signal"] = _generate_signals(
        condition_buy=data["Close"] > tsf, condition_sell=data["Close"] < tsf
    )
    return data["TSF_Signal"]


def VAR_indicator(data, timeperiod=5, nbdev=1):
    """Vectorized Variance (VAR) indicator signals."""
    var = ta.VAR(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    data["VAR_Signal"] = _generate_signals(
        condition_buy=var > 20, condition_sell=var < 10
    )
    logger.warning(
        "Warning: Using fixed VAR levels (10, 20) for Buy/Sell signals in VAR_indicator is unconventional."
    )
    return data["VAR_Signal"]


# --- New Indicator Functions ---


def ichimoku_cloud_indicator(
    data, period_tenkan=9, period_kijun=26, period_senkou_b=52
):
    """
    Calculates Ichimoku Cloud components and adds them to the DataFrame.
    Also adds a basic Price vs Cloud signal.

    Components Added:
    - Ichi_Tenkan: Tenkan-sen (Conversion Line)
    - Ichi_Kijun: Kijun-sen (Base Line)
    - Ichi_SenkouA: Senkou Span A (Leading Span A)
    - Ichi_SenkouB: Senkou Span B (Leading Span B)
    - Ichi_Chikou: Chikou Span (Lagging Span)
    - Ichimoku_Signal: Basic signal based on Price position relative to the Cloud.

    Standard Periods: tenkan=9, kijun=26, senkou_b=52. Kijun period is also used for shifts.
    """
    required_cols = ["High", "Low", "Close"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")

    high_prices = data["High"]
    low_prices = data["Low"]
    close_prices = data["Close"]

    nine_period_high = high_prices.rolling(window=period_tenkan).max()
    nine_period_low = low_prices.rolling(window=period_tenkan).min()
    data["Ichi_Tenkan"] = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = high_prices.rolling(window=period_kijun).max()
    twenty_six_period_low = low_prices.rolling(window=period_kijun).min()
    data["Ichi_Kijun"] = (twenty_six_period_high + twenty_six_period_low) / 2

    data["Ichi_SenkouA"] = ((data["Ichi_Tenkan"] + data["Ichi_Kijun"]) / 2).shift(
        period_kijun
    )

    fifty_two_period_high = high_prices.rolling(window=period_senkou_b).max()
    fifty_two_period_low = low_prices.rolling(window=period_senkou_b).min()
    data["Ichi_SenkouB"] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(
        period_kijun
    )

    data["Ichi_Chikou"] = close_prices.shift(-period_kijun)

    above_cloud = (close_prices > data["Ichi_SenkouA"]) & (
        close_prices > data["Ichi_SenkouB"]
    )
    below_cloud = (close_prices < data["Ichi_SenkouA"]) & (
        close_prices < data["Ichi_SenkouB"]
    )

    data["Ichimoku_Signal"] = _generate_signals(
        condition_buy=above_cloud, condition_sell=below_cloud
    )
    data.drop(
        columns=[
            "Ichi_Tenkan",
            "Ichi_Kijun",
            "Ichi_SenkouA",
            "Ichi_SenkouB",
            "Ichi_Chikou",
        ],
        inplace=True,
    )
    return data["Ichimoku_Signal"]


def keltner_channels_indicator(data, period_ema=20, period_atr=10, multiplier=2.0):
    """
    Calculates Keltner Channels and adds them to the DataFrame.
    Also adds a basic channel breakout signal.

    Components Added:
    - KC_Middle: Middle Line (EMA)
    - KC_Upper: Upper Keltner Channel
    - KC_Lower: Lower Keltner Channel
    - Keltner_Signal: Basic signal based on Price breaking outside the channels.
    """
    required_cols = ["High", "Low", "Close"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")

    data["KC_Middle"] = ta.EMA(data["Close"], timeperiod=period_ema)

    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=period_atr)

    data["KC_Upper"] = data["KC_Middle"] + (multiplier * atr)
    data["KC_Lower"] = data["KC_Middle"] - (multiplier * atr)

    data["Keltner_Signal"] = _generate_signals(
        condition_buy=data["Close"] > data["KC_Upper"],
        condition_sell=data["Close"] < data["KC_Lower"],
    )
    data.drop(columns=["KC_Middle", "KC_Upper", "KC_Lower"], inplace=True)
    return data["Keltner_Signal"]


def vwap_indicator(data, window=14):
    """
    Calculates a rolling Volume Weighted Average Price (VWAP) and adds it to the DataFrame.
    Also adds a basic Price vs VWAP signal.

    Note: This is a ROLLING VWAP over the specified window. For intraday trading,
    VWAP is often reset daily. This requires different logic (groupby date).

    Components Added:
    - VWAP: Rolling VWAP value
    - VWAP_Signal: Basic signal based on Close price relative to VWAP.
    """
    required_cols = ["High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")
    if data["Volume"].isnull().any() or (data["Volume"] < 0).any():
        logger.warning(
            "Warning: VWAP calculation encountered missing or negative Volume data. Results may be inaccurate."
        )

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3

    tp_vol = typical_price * data["Volume"]
    sum_tp_vol = tp_vol.rolling(window=window, min_periods=window).sum()
    sum_vol = data["Volume"].rolling(window=window, min_periods=window).sum()

    sum_vol_safe = sum_vol.replace(0, np.nan)
    data["VWAP"] = sum_tp_vol / sum_vol_safe
    # data["VWAP"] = data["VWAP"].fillna(method="ffill")
    data["VWAP"] = data["VWAP"].ffill()

    data["VWAP_Signal"] = _generate_signals(
        condition_buy=data["Close"] > data["VWAP"],
        condition_sell=data["Close"] < data["VWAP"],
    )

    data.drop(columns=["VWAP"], inplace=True)

    return data["VWAP_Signal"]


# --- Example Usage ---
if __name__ == "__main__":
    ...
