import pandas as pd
import talib as ta
import numpy as np

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
    # Note: Original logic was Close > Upper = Sell, Close < Lower = Buy
    data["BBANDS_Signal"] = _generate_signals(
        condition_buy=data["Close"] < lower, condition_sell=data["Close"] > upper
    )
    # Optionally, add the bands themselves
    # data['BB_Upper'] = upper
    # data['BB_Middle'] = middle
    # data['BB_Lower'] = lower
    return data


def DEMA_indicator(data, timeperiod=30):
    """Vectorized Double Exponential Moving Average (DEMA) indicator signals."""
    dema = ta.DEMA(data["Close"], timeperiod=timeperiod)
    data["DEMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > dema, condition_sell=data["Close"] < dema
    )
    # data['DEMA'] = dema
    return data


def EMA_indicator(data, timeperiod=30):
    """Vectorized Exponential Moving Average (EMA) indicator signals."""
    ema = ta.EMA(data["Close"], timeperiod=timeperiod)
    data["EMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ema, condition_sell=data["Close"] < ema
    )
    # data['EMA'] = ema
    return data


def HT_TRENDLINE_indicator(data):
    """Vectorized Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) signals."""
    ht_trendline = ta.HT_TRENDLINE(data["Close"])
    data["HT_TRENDLINE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ht_trendline,
        condition_sell=data["Close"] < ht_trendline,
    )
    # data['HT_TRENDLINE'] = ht_trendline
    return data


def KAMA_indicator(data, timeperiod=30):
    """Vectorized Kaufman Adaptive Moving Average (KAMA) indicator signals."""
    kama = ta.KAMA(data["Close"], timeperiod=timeperiod)
    data["KAMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > kama, condition_sell=data["Close"] < kama
    )
    # data['KAMA'] = kama
    return data


def MA_indicator(data, timeperiod=30, matype=0):
    """Vectorized Moving average (MA) indicator signals."""
    ma = ta.MA(data["Close"], timeperiod=timeperiod, matype=matype)
    data["MA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > ma, condition_sell=data["Close"] < ma
    )
    # data['MA'] = ma
    return data


def MAMA_indicator(data, fastlimit=0.5, slowlimit=0.05):
    """Vectorized MESA Adaptive Moving Average (MAMA) indicator signals."""
    # Need to use .values for MAMA/FAMA with Series input if ta-lib version requires
    try:
        # Try with Series first (newer ta-lib versions might handle it)
        mama, fama = ta.MAMA(data["Close"], fastlimit=fastlimit, slowlimit=slowlimit)
    except:
        # Fallback to numpy array if Series input fails
        mama_vals, fama_vals = ta.MAMA(
            data["Close"].values, fastlimit=fastlimit, slowlimit=slowlimit
        )
        # Convert back to Series, aligning index with original data
        mama = pd.Series(mama_vals, index=data.index)
        fama = pd.Series(fama_vals, index=data.index)

    data["MAMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > mama, condition_sell=data["Close"] < mama
    )
    # data['MAMA'] = mama
    # data['FAMA'] = fama
    return data


def MAVP_indicator(data, minperiod=2, maxperiod=30, matype=0):
    """
    Vectorized Moving Average with Variable Period (MAVP) indicator signals.
    Note: MAVP requires a 'periods' array/Series. You must ensure a column named
    'periods' (or pass it explicitly) exists in the DataFrame or is created before calling.
    This example creates a constant period for demonstration.
    """
    # Check if a 'periods' column exists, otherwise create a default constant one
    if "periods" not in data.columns:
        print(
            "Warning: 'periods' column not found for MAVP_indicator. Creating a constant period Series (30.0)."
        )
        # Example: Create a simple constant periods Series
        periods_series = pd.Series(30.0, index=data.index)  # Must be float

        periods_input = periods_series
    else:
        periods_input = data["periods"].astype(float)  # Ensure float type

    # Ensure periods length matches data length (handle NaNs if necessary)
    periods_aligned = (
        periods_input.reindex(data.index).fillna(method="ffill").fillna(30.0)
    )  # Example fill logic

    # Need to use .values for MAVP with Series input if ta-lib version requires
    try:
        # Try with Series first
        mavp = ta.MAVP(
            data["Close"],
            periods=periods_aligned,
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=matype,
        )
    except TypeError:  # Catch if it specifically requires ndarray
        # Fallback to numpy array if Series input fails
        mavp_vals = ta.MAVP(
            data["Close"].values,
            periods=periods_aligned.values,
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=matype,
        )
        mavp = pd.Series(mavp_vals, index=data.index)  # Convert back to Series

    data["MAVP_Signal"] = _generate_signals(
        condition_buy=data["Close"] > mavp, condition_sell=data["Close"] < mavp
    )
    # data['MAVP'] = mavp
    return data


def MIDPOINT_indicator(data, timeperiod=14):
    """Vectorized MidPoint over period (MIDPOINT) indicator signals."""
    midpoint = ta.MIDPOINT(data["Close"], timeperiod=timeperiod)
    data["MIDPOINT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > midpoint, condition_sell=data["Close"] < midpoint
    )
    # data['MIDPOINT'] = midpoint
    return data


def MIDPRICE_indicator(data, timeperiod=14):
    """Vectorized Midpoint Price over period (MIDPRICE) indicator signals."""
    midprice = ta.MIDPRICE(data["High"], data["Low"], timeperiod=timeperiod)
    data["MIDPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > midprice, condition_sell=data["Close"] < midprice
    )
    # data['MIDPRICE'] = midprice
    return data


def SAR_indicator(data, acceleration=0.02, maximum=0.2):  # Using common defaults
    """Vectorized Parabolic SAR (SAR) indicator signals."""
    # Note: Original used acceleration=0, maximum=0. Using common defaults instead.
    sar = ta.SAR(data["High"], data["Low"], acceleration=acceleration, maximum=maximum)
    data["SAR_Signal"] = _generate_signals(
        condition_buy=data["Close"] > sar, condition_sell=data["Close"] < sar
    )
    # data['SAR'] = sar
    return data


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
):  # Using common defaults
    """Vectorized Parabolic SAR - Extended (SAREXT) indicator signals."""
    # Note: Original used all acceleration params = 0. Using common defaults instead.
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
    # data['SAREXT'] = sarext
    return data


def SMA_indicator(data, timeperiod=30):
    """Vectorized Simple Moving Average (SMA) indicator signals."""
    sma = ta.SMA(data["Close"], timeperiod=timeperiod)
    data["SMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > sma, condition_sell=data["Close"] < sma
    )
    # data['SMA'] = sma
    return data


def T3_indicator(data, timeperiod=5, vfactor=0.7):  # Using common defaults
    """Vectorized Triple Exponential Moving Average (T3) indicator signals."""
    # Note: Original used timeperiod=30, vfactor=0. Using common defaults: timeperiod=5, vfactor=0.7
    t3 = ta.T3(data["Close"], timeperiod=timeperiod, vfactor=vfactor)
    data["T3_Signal"] = _generate_signals(
        condition_buy=data["Close"] > t3, condition_sell=data["Close"] < t3
    )
    # data['T3'] = t3
    return data


def TEMA_indicator(data, timeperiod=30):
    """Vectorized Triple Exponential Moving Average (TEMA) indicator signals."""
    tema = ta.TEMA(data["Close"], timeperiod=timeperiod)
    data["TEMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > tema, condition_sell=data["Close"] < tema
    )
    # data['TEMA'] = tema
    return data


def TRIMA_indicator(data, timeperiod=30):
    """Vectorized Triangular Moving Average (TRIMA) indicator signals."""
    trima = ta.TRIMA(data["Close"], timeperiod=timeperiod)
    data["TRIMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > trima, condition_sell=data["Close"] < trima
    )
    # data['TRIMA'] = trima
    return data


def WMA_indicator(data, timeperiod=30):
    """Vectorized Weighted Moving Average (WMA) indicator signals."""
    wma = ta.WMA(data["Close"], timeperiod=timeperiod)
    data["WMA_Signal"] = _generate_signals(
        condition_buy=data["Close"] > wma, condition_sell=data["Close"] < wma
    )
    # data['WMA'] = wma
    return data


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

    # Crossover conditions
    di_cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    di_cross_down = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))

    # Trend strength condition
    is_trending = adx > adx_threshold

    # Generate signals: Buy on DI+ cross up IF trending, Sell on DI- cross up IF trending
    # Using np.select for clarity on conditions
    conditions = [
        (di_cross_up) & is_trending,  # DI+ is dominant and trend is strong
        (di_cross_down) & is_trending,  # DI- is dominant and trend is strong
        # (plus_di > minus_di) & is_trending,  # DI+ is dominant and trend is strong
        # (minus_di > plus_di) & is_trending,  # DI- is dominant and trend is strong
    ]
    choices = ["Buy", "Sell"]
    data["ADX_Signal"] = np.select(conditions, choices, default="Hold")

    # Optional simpler crossover without ADX filter:
    # data['ADX_Signal_Unfiltered'] = np.where(plus_di > minus_di, 'Buy', 'Sell')

    # data['ADX'] = adx
    # data['PLUS_DI'] = plus_di
    # data['MINUS_DI'] = minus_di
    return data


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

    # Trend strength condition (using ADXR)
    is_trending = adxr > adx_threshold

    # Generate signals based on DI dominance IF trending
    conditions = [
        (plus_di > minus_di) & is_trending,  # DI+ dominant & trending
        (minus_di > plus_di) & is_trending,  # DI- dominant & trending
    ]
    choices = ["Buy", "Sell"]
    data["ADXR_Signal"] = np.select(conditions, choices, default="Hold")

    # data['ADXR'] = adxr
    # data['PLUS_DI'] = plus_di
    # data['MINUS_DI'] = minus_di
    print(
        "Warning: Filtering signals based on ADXR > threshold is less common than using ADX."
    )
    return data


def CCI_indicator_v2(data, timeperiod=14, buy_level=-100, sell_level=100):
    """
    Vectorized Commodity Channel Index (CCI) indicator signals.
    Standard interpretation: Buy when crossing UP from oversold (< buy_level),
    Sell when crossing DOWN from overbought (> sell_level).
    Simplified version: Buy if < buy_level, Sell if > sell_level.
    """
    cci = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

    # Revised logic: Buy when oversold, Sell when overbought (reversal)
    data["CCI_Signal"] = _generate_signals(
        condition_buy=cci < buy_level,  # Oversold condition
        condition_sell=cci > sell_level,  # Overbought condition
    )
    # data['CCI'] = cci
    return data


def CMO_indicator_v2(data, timeperiod=14, buy_level=-50, sell_level=50):
    """
    Vectorized Chande Momentum Oscillator (CMO) indicator signals.
    Standard interpretation: Buy when oversold (< buy_level),
    Sell when overbought (> sell_level).
    """
    cmo = ta.CMO(data["Close"], timeperiod=timeperiod)

    # Revised logic: Buy when oversold, Sell when overbought (reversal)
    data["CMO_Signal"] = _generate_signals(
        condition_buy=cmo < buy_level,  # Oversold condition
        condition_sell=cmo > sell_level,  # Overbought condition
    )
    # data['CMO'] = cmo
    return data


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

    # Trend strength condition (using DX)
    is_trending = dx > dx_threshold

    # Generate signals based on DI dominance IF trending
    conditions = [
        (plus_di > minus_di) & is_trending,  # DI+ dominant & trending
        (minus_di > plus_di) & is_trending,  # DI- dominant & trending
    ]
    choices = ["Buy", "Sell"]
    data["DX_Signal"] = np.select(conditions, choices, default="Hold")

    # data['DX'] = dx
    # data['PLUS_DI'] = plus_di
    # data['MINUS_DI'] = minus_di
    return data


# MINUS_DI and PLUS_DI indicators based on simple dominance (crossover implied)
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
        condition_buy=plus_di > minus_di,  # DI+ is dominant
        condition_sell=minus_di > plus_di,  # DI- is dominant
    )
    # data['MINUS_DI'] = minus_di
    # data['PLUS_DI'] = plus_di
    return data


# --- Momentum Indicators ---


# superseeded
def ADX_indicator(data, timeperiod=14):
    """Vectorized Average Directional Movement Index (ADX) indicator signals."""
    adx = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Original logic: > 25 Buy, < 20 Sell. This represents trend strength, not direction usually.
    data["ADX_Signal"] = _generate_signals(
        condition_buy=adx > 25, condition_sell=adx < 20
    )
    # data['ADX'] = adx
    return data


# superseeded
def ADXR_indicator(data, timeperiod=14):
    """Vectorized Average Directional Movement Index Rating (ADXR) indicator signals."""
    adxr = ta.ADXR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Similar comment as ADX regarding the logic.
    data["ADXR_Signal"] = _generate_signals(
        condition_buy=adxr > 25, condition_sell=adxr < 20
    )
    # data['ADXR'] = adxr
    return data


def APO_indicator(data, fastperiod=12, slowperiod=26, matype=0):
    """Vectorized Absolute Price Oscillator (APO) indicator signals."""
    apo = ta.APO(
        data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype
    )
    data["APO_Signal"] = _generate_signals(
        condition_buy=apo > 0, condition_sell=apo < 0
    )
    # data['APO'] = apo
    return data


def AROON_indicator(data, timeperiod=14):
    """Vectorized Aroon (AROON) indicator signals."""
    aroon_down, aroon_up = ta.AROON(data["High"], data["Low"], timeperiod=timeperiod)
    # Original logic: aroon_up > 70 Buy, aroon_down > 70 Sell
    data["AROON_Signal"] = _generate_signals(
        condition_buy=aroon_up > 70, condition_sell=aroon_down > 70
    )
    # data['Aroon_Down'] = aroon_down
    # data['Aroon_Up'] = aroon_up
    return data


def AROONOSC_indicator(data, timeperiod=14):
    """Vectorized Aroon Oscillator (AROONOSC) indicator signals."""
    aroonosc = ta.AROONOSC(data["High"], data["Low"], timeperiod=timeperiod)
    data["AROONOSC_Signal"] = _generate_signals(
        condition_buy=aroonosc > 0, condition_sell=aroonosc < 0
    )
    # data['AROONOSC'] = aroonosc
    return data


def BOP_indicator(data):
    """Vectorized Balance Of Power (BOP) indicator signals."""
    bop = ta.BOP(data["Open"], data["High"], data["Low"], data["Close"])
    data["BOP_Signal"] = _generate_signals(
        condition_buy=bop > 0, condition_sell=bop < 0
    )
    # data['BOP'] = bop
    return data


# superseeded
def CCI_indicator(data, timeperiod=14):
    """Vectorized Commodity Channel Index (CCI) indicator signals."""
    cci = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Sticking to original request's logic: > 100 Buy, < -100 Sell
    data["CCI_Signal"] = _generate_signals(
        condition_buy=cci > 100, condition_sell=cci < -100
    )
    # data['CCI'] = cci
    return data


# superseeded
def CMO_indicator(data, timeperiod=14):
    """Vectorized Chande Momentum Oscillator (CMO) indicator signals."""
    cmo = ta.CMO(data["Close"], timeperiod=timeperiod)
    # Sticking to original request's logic: > 50 Buy, < -50 Sell
    data["CMO_Signal"] = _generate_signals(
        condition_buy=cmo > 50, condition_sell=cmo < -50
    )
    # data['CMO'] = cmo
    return data


# superseeded
def DX_indicator(data, timeperiod=14):
    """Vectorized Directional Movement Index (DX) indicator signals."""
    dx = ta.DX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Original logic: > 25 Buy, < 20 Sell. Like ADX, usually indicates strength.
    data["DX_Signal"] = _generate_signals(condition_buy=dx > 25, condition_sell=dx < 20)
    # data['DX'] = dx
    return data


def MACD_indicator(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """Vectorized Moving Average Convergence/Divergence (MACD) indicator signals."""
    macd, macdsignal, macdhist = ta.MACD(
        data["Close"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    # Signal based on histogram (MACD line vs Signal line)
    data["MACD_Signal"] = _generate_signals(
        condition_buy=macdhist > 0,  # MACD above signal line
        condition_sell=macdhist < 0,  # MACD below signal line
    )
    # data['MACD'] = macd
    # data['MACD_Signal_Line'] = macdsignal
    # data['MACD_Hist'] = macdhist
    return data


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
    # data['MACDEXT'] = macd
    # data['MACDEXT_Signal_Line'] = macdsignal
    # data['MACDEXT_Hist'] = macdhist
    return data


def MACDFIX_indicator(data, signalperiod=9):
    """Vectorized Moving Average Convergence/Divergence Fix 12/26 (MACDFIX) signals."""
    macd, macdsignal, macdhist = ta.MACDFIX(data["Close"], signalperiod=signalperiod)
    data["MACDFIX_Signal"] = _generate_signals(
        condition_buy=macdhist > 0, condition_sell=macdhist < 0
    )
    # data['MACDFIX'] = macd
    # data['MACDFIX_Signal_Line'] = macdsignal
    # data['MACDFIX_Hist'] = macdhist
    return data


def MFI_indicator(data, timeperiod=14):
    """Vectorized Money Flow Index (MFI) indicator signals."""
    # Check if 'Volume' column exists
    if "Volume" not in data.columns:
        raise ValueError("MFI_indicator requires 'Volume' column in data")

    mfi = ta.MFI(
        data["High"], data["Low"], data["Close"], data["Volume"], timeperiod=timeperiod
    )
    # Original logic: > 80 Sell (Overbought), < 20 Buy (Oversold)
    data["MFI_Signal"] = _generate_signals(
        condition_buy=mfi < 20, condition_sell=mfi > 80
    )
    # data['MFI'] = mfi
    return data


# superseeded -combined
def MINUS_DI_indicator(data, timeperiod=14):
    """Vectorized Minus Directional Indicator (MINUS_DI) indicator signals."""
    minus_di = ta.MINUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    # Original logic: > 25 Sell, < 20 Buy. DI- measures downward pressure.
    data["MINUS_DI_Signal"] = _generate_signals(
        condition_buy=minus_di < 20,  # Low downward pressure
        condition_sell=minus_di > 25,  # High downward pressure
    )
    # data['MINUS_DI'] = minus_di
    return data


# superseeded -combined
def PLUS_DI_indicator(data, timeperiod=14):
    """Vectorized Plus Directional Indicator (PLUS_DI) indicator signals."""
    plus_di = ta.PLUS_DI(
        data["High"], data["Low"], data["Close"], timeperiod=timeperiod
    )
    # Original logic: > 25 Buy, < 20 Sell. DI+ measures upward pressure.
    data["PLUS_DI_Signal"] = _generate_signals(
        condition_buy=plus_di > 25,  # High upward pressure
        condition_sell=plus_di < 20,  # Low upward pressure
    )
    # data['PLUS_DI'] = plus_di
    return data


def MINUS_DM_indicator(data, timeperiod=14):
    """Vectorized Minus Directional Movement (MINUS_DM) indicator signals."""
    minus_dm = ta.MINUS_DM(data["High"], data["Low"], timeperiod=timeperiod)
    # Original logic: > 0 Sell, < 0 Buy. DM is usually >= 0. This logic seems incorrect.
    data["MINUS_DM_Signal"] = _generate_signals(
        condition_buy=minus_dm < 0,  # This condition will likely never be true
        condition_sell=minus_dm > 0,
    )
    # data['MINUS_DM'] = minus_dm
    # **WARNING**: The original logic for MINUS_DM seems flawed. Review required.
    print(
        "Warning: The implemented logic for MINUS_DM_indicator based on the original function seems potentially incorrect."
    )
    return data


def MOM_indicator(data, timeperiod=10):
    """Vectorized Momentum (MOM) indicator signals."""
    mom = ta.MOM(data["Close"], timeperiod=timeperiod)
    data["MOM_Signal"] = _generate_signals(
        condition_buy=mom > 0, condition_sell=mom < 0
    )
    # data['MOM'] = mom
    return data


def PLUS_DM_indicator(data, timeperiod=14):
    """Vectorized Plus Directional Movement (PLUS_DM) indicator signals."""
    plus_dm = ta.PLUS_DM(data["High"], data["Low"], timeperiod=timeperiod)
    # Original logic: > 0 Buy, < 0 Sell. DM is usually >= 0. This logic seems incorrect.
    data["PLUS_DM_Signal"] = _generate_signals(
        condition_buy=plus_dm > 0,
        condition_sell=plus_dm < 0,  # This condition will likely never be true
    )
    # data['PLUS_DM'] = plus_dm
    # **WARNING**: The original logic for PLUS_DM seems flawed. Review required.
    print(
        "Warning: The implemented logic for PLUS_DM_indicator based on the original function seems potentially incorrect."
    )
    return data


def PPO_indicator(data, fastperiod=12, slowperiod=26, matype=0):
    """Vectorized Percentage Price Oscillator (PPO) indicator signals."""
    ppo = ta.PPO(
        data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype
    )
    data["PPO_Signal"] = _generate_signals(
        condition_buy=ppo > 0, condition_sell=ppo < 0
    )
    # data['PPO'] = ppo
    return data


def ROC_indicator(data, timeperiod=10):
    """Vectorized Rate of change : ((price/prevPrice)-1)*100 (ROC) signals."""
    roc = ta.ROC(data["Close"], timeperiod=timeperiod)
    data["ROC_Signal"] = _generate_signals(
        condition_buy=roc > 0, condition_sell=roc < 0
    )
    # data['ROC'] = roc
    return data


def ROCP_indicator(data, timeperiod=10):
    """Vectorized Rate of change Percentage: (price-prevPrice)/prevPrice signals."""
    rocp = ta.ROCP(data["Close"], timeperiod=timeperiod)
    data["ROCP_Signal"] = _generate_signals(
        condition_buy=rocp > 0, condition_sell=rocp < 0
    )
    # data['ROCP'] = rocp
    return data


def ROCR_indicator(data, timeperiod=10):
    """Vectorized Rate of change ratio: (price/prevPrice) signals."""
    rocr = ta.ROCR(data["Close"], timeperiod=timeperiod)
    data["ROCR_Signal"] = _generate_signals(
        condition_buy=rocr > 1, condition_sell=rocr < 1
    )
    # data['ROCR'] = rocr
    return data


def ROCR100_indicator(data, timeperiod=10):
    """Vectorized Rate of change ratio 100 scale: (price/prevPrice)*100 signals."""
    rocr100 = ta.ROCR100(data["Close"], timeperiod=timeperiod)
    data["ROCR100_Signal"] = _generate_signals(
        condition_buy=rocr100 > 100, condition_sell=rocr100 < 100
    )
    # data['ROCR100'] = rocr100
    return data


def RSI_indicator(data, timeperiod=14):
    """Vectorized Relative Strength Index (RSI) indicator signals."""
    rsi = ta.RSI(data["Close"], timeperiod=timeperiod)
    # Original logic: > 70 Sell (Overbought), < 30 Buy (Oversold)
    data["RSI_Signal"] = _generate_signals(
        condition_buy=rsi < 30, condition_sell=rsi > 70
    )
    # data['RSI'] = rsi
    return data


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
    # Original logic: SlowK > 80 Sell (Overbought), SlowK < 20 Buy (Oversold)
    data["STOCH_Signal"] = _generate_signals(
        condition_buy=slowk < 20, condition_sell=slowk > 80
    )
    # data['STOCH_SlowK'] = slowk
    # data['STOCH_SlowD'] = slowd
    return data


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
    # Original logic: FastK > 80 Sell (Overbought), FastK < 20 Buy (Oversold)
    data["STOCHF_Signal"] = _generate_signals(
        condition_buy=fastk < 20, condition_sell=fastk > 80
    )
    # data['STOCHF_FastK'] = fastk
    # data['STOCHF_FastD'] = fastd
    return data


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
    # Original logic: FastK > 80 Sell (Overbought), FastK < 20 Buy (Oversold)
    data["STOCHRSI_Signal"] = _generate_signals(
        condition_buy=fastk < 20, condition_sell=fastk > 80
    )
    # data['STOCHRSI_FastK'] = fastk
    # data['STOCHRSI_FastD'] = fastd
    return data


def TRIX_indicator(data, timeperiod=30):
    """Vectorized 1-day ROC of a Triple Smooth EMA (TRIX) indicator signals."""
    trix = ta.TRIX(data["Close"], timeperiod=timeperiod)
    # Signal based on TRIX crossing zero
    data["TRIX_Signal"] = _generate_signals(
        condition_buy=trix > 0, condition_sell=trix < 0
    )
    # data['TRIX'] = trix
    return data


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
    # Original logic: > 70 Sell (Overbought), < 30 Buy (Oversold)
    data["ULTOSC_Signal"] = _generate_signals(
        condition_buy=ultosc < 30, condition_sell=ultosc > 70
    )
    # data['ULTOSC'] = ultosc
    return data


def WILLR_indicator(data, timeperiod=14):
    """Vectorized Williams' %R (WILLR) indicator signals."""
    willr = ta.WILLR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Original logic: > -20 Sell (Overbought), < -80 Buy (Oversold)
    data["WILLR_Signal"] = _generate_signals(
        condition_buy=willr < -80, condition_sell=willr > -20
    )
    # data['WILLR'] = willr
    return data


# --- Revised Volume Indicators ---


def AD_indicator_v2(data, ma_period=20):
    """
    Vectorized Chaikin A/D Line (AD) indicator signals.
    Revised logic: Compare AD line to its moving average.
    """
    if "Volume" not in data.columns:
        raise ValueError("AD_indicator requires 'Volume' column in data")
    ad = ta.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    ad_ma = ta.SMA(ad, timeperiod=ma_period)  # Calculate moving average of AD line

    data["AD_Signal"] = _generate_signals(
        condition_buy=ad > ad_ma,  # AD line above its average
        condition_sell=ad < ad_ma,  # AD line below its average
    )
    # data['AD'] = ad
    # data['AD_MA'] = ad_ma
    print("Note: AD_indicator revised logic uses AD crossing its SMA.")
    return data


def OBV_indicator_v2(data, ma_period=20):
    """
    Vectorized On Balance Volume (OBV) indicator signals.
    Revised logic: Compare OBV to its moving average.
    """
    if "Volume" not in data.columns:
        raise ValueError("OBV_indicator requires 'Volume' column in data")
    obv = ta.OBV(data["Close"], data["Volume"])
    obv_ma = ta.SMA(obv, timeperiod=ma_period)  # Calculate moving average of OBV

    data["OBV_Signal"] = _generate_signals(
        condition_buy=obv > obv_ma,  # OBV above its average
        condition_sell=obv < obv_ma,  # OBV below its average
    )
    # data['OBV'] = obv
    # data['OBV_MA'] = obv_ma
    print("Note: OBV_indicator revised logic uses OBV crossing its SMA.")
    return data


# --- Volume Indicators ---


def AD_indicator(data):
    """Vectorized Chaikin A/D Line (AD) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("AD_indicator requires 'Volume' column in data")
    ad = ta.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    # Original logic: > 0 Buy, < 0 Sell. This is unusual for A/D line itself.
    data["AD_Signal"] = _generate_signals(condition_buy=ad > 0, condition_sell=ad < 0)
    # data['AD'] = ad
    # **WARNING**: The original logic for AD seems potentially unconventional. Review required.
    print(
        "Warning: The implemented logic for AD_indicator based on the original function might be unconventional."
    )
    return data


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
    # Signal based on crossing zero line
    data["ADOSC_Signal"] = _generate_signals(
        condition_buy=adosc > 0, condition_sell=adosc < 0
    )
    # data['ADOSC'] = adosc
    return data


def OBV_indicator(data):
    """Vectorized On Balance Volume (OBV) indicator signals."""
    if "Volume" not in data.columns:
        raise ValueError("OBV_indicator requires 'Volume' column in data")
    obv = ta.OBV(data["Close"], data["Volume"])
    # Original logic: > 0 Buy, < 0 Sell. OBV value itself isn't typically compared to 0.
    data["OBV_Signal"] = _generate_signals(
        condition_buy=obv > 0, condition_sell=obv < 0
    )
    # data['OBV'] = obv
    # **WARNING**: The original logic for OBV seems potentially unconventional. Review required.
    print(
        "Warning: The implemented logic for OBV_indicator based on the original function might be unconventional."
    )
    return data


# --- Revised Cycle Indicators ---


def HT_TRENDMODE_indicator_v2(data):
    """
    Vectorized Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) signals.
    Revised logic: Buy in Trend Mode (1), Sell in Cycle Mode (0).
    """
    ht_trendmode = ta.HT_TRENDMODE(
        data["Close"]
    )  # Returns 1 for trend mode, 0 for cycle mode

    data["HT_TRENDMODE_Signal"] = _generate_signals(
        condition_buy=ht_trendmode == 1,  # Trend mode
        condition_sell=ht_trendmode == 0,  # Cycle mode
    )
    # data['HT_TRENDMODE'] = ht_trendmode
    print(
        "Note: HT_TRENDMODE_indicator revised logic: Buy on Trend(1), Sell on Cycle(0)."
    )
    return data


# --- Cycle Indicators ---


def HT_DCPERIOD_indicator(data):
    """Vectorized Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD) signals."""
    ht_dcperiod = ta.HT_DCPERIOD(data["Close"])
    # Original logic: > 20 Buy, < 10 Sell. Interpreting cycle length.
    data["HT_DCPERIOD_Signal"] = _generate_signals(
        condition_buy=ht_dcperiod > 20,  # Longer cycle dominance?
        condition_sell=ht_dcperiod < 10,  # Shorter cycle dominance?
    )
    # data['HT_DCPERIOD'] = ht_dcperiod
    return data


def HT_DCPHASE_indicator(data):
    """Vectorized Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) signals."""
    ht_dcphase = ta.HT_DCPHASE(data["Close"])
    # Original logic: > 0 Buy, < 0 Sell. Phase interpretation.
    data["HT_DCPHASE_Signal"] = _generate_signals(
        condition_buy=ht_dcphase > 0, condition_sell=ht_dcphase < 0
    )
    # data['HT_DCPHASE'] = ht_dcphase
    return data


def HT_PHASOR_indicator(data):
    """Vectorized Hilbert Transform - Phasor Components (HT_PHASOR) signals (using inphase)."""
    inphase, quadrature = ta.HT_PHASOR(data["Close"])
    # Original logic based on inphase component
    data["HT_PHASOR_Signal"] = _generate_signals(
        condition_buy=inphase > 0, condition_sell=inphase < 0
    )
    # data['HT_InPhase'] = inphase
    # data['HT_Quadrature'] = quadrature
    return data


def HT_SINE_indicator(data):
    """Vectorized Hilbert Transform - SineWave (HT_SINE) indicator signals (using sine)."""
    sine, leadsine = ta.HT_SINE(data["Close"])
    # Original logic based on sine wave position
    data["HT_SINE_Signal"] = _generate_signals(
        condition_buy=sine > 0,  # Rising phase?
        condition_sell=sine < 0,  # Falling phase?
    )
    # data['HT_Sine'] = sine
    # data['HT_LeadSine'] = leadsine
    return data


# superseeded
def HT_TRENDMODE_indicator(data):
    """Vectorized Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) signals."""
    ht_trendmode = ta.HT_TRENDMODE(
        data["Close"]
    )  # Returns 1 for trend mode, 0 for cycle mode
    # Implementing original logic as requested (though potentially flawed):
    data["HT_TRENDMODE_Signal"] = _generate_signals(
        condition_buy=ht_trendmode > 0,  # Trend mode (== 1)
        condition_sell=ht_trendmode < 0,  # Never happens
    )
    # data['HT_TRENDMODE'] = ht_trendmode
    print(
        "Warning: Original HT_TRENDMODE_indicator logic might be flawed (Sell condition never met)."
    )
    return data


# --- Price Transform ---


def AVGPRICE_indicator(data):
    """Vectorized Average Price (AVGPRICE) indicator signals."""
    avgprice = ta.AVGPRICE(data["Open"], data["High"], data["Low"], data["Close"])
    data["AVGPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > avgprice, condition_sell=data["Close"] < avgprice
    )
    # data['AVGPRICE'] = avgprice
    return data


def MEDPRICE_indicator(data):
    """Vectorized Median Price (MEDPRICE) indicator signals."""
    medprice = ta.MEDPRICE(data["High"], data["Low"])
    data["MEDPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > medprice, condition_sell=data["Close"] < medprice
    )
    # data['MEDPRICE'] = medprice
    return data


def TYPPRICE_indicator(data):
    """Vectorized Typical Price (TYPPRICE) indicator signals."""
    typprice = ta.TYPPRICE(data["High"], data["Low"], data["Close"])
    data["TYPPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > typprice, condition_sell=data["Close"] < typprice
    )
    # data['TYPPRICE'] = typprice
    return data


def WCLPRICE_indicator(data):
    """Vectorized Weighted Close Price (WCLPRICE) indicator signals."""
    wclprice = ta.WCLPRICE(data["High"], data["Low"], data["Close"])
    data["WCLPRICE_Signal"] = _generate_signals(
        condition_buy=data["Close"] > wclprice, condition_sell=data["Close"] < wclprice
    )
    # data['WCLPRICE'] = wclprice
    return data


# --- Revised Volatility Indicators ---
# WARNING: Using volatility indicators directly for Buy/Sell is generally not recommended.
# They are better used for risk management or confirming other signals.
# The MA crossover logic provided is a *possible* interpretation but lacks strong theoretical backing as a primary signal.


def ATR_indicator_v2(data, timeperiod=14, ma_period=14):
    """
    Vectorized Average True Range (ATR) indicator signals.
    Revised logic: Compare ATR to its moving average.
    WARNING: This is not a standard signal generation technique for ATR.
    """
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    atr_ma = ta.SMA(atr, timeperiod=ma_period)

    data["ATR_Signal"] = _generate_signals(
        condition_buy=atr > atr_ma,  # Volatility increasing / above average
        condition_sell=atr < atr_ma,  # Volatility decreasing / below average
    )
    # data['ATR'] = atr
    # data['ATR_MA'] = atr_ma
    print(
        "Warning: ATR_indicator revised logic (ATR vs MA) is unconventional for Buy/Sell signals."
    )
    return data


def NATR_indicator_v2(data, timeperiod=14, ma_period=14):
    """
    Vectorized Normalized Average True Range (NATR) indicator signals.
    Revised logic: Compare NATR to its moving average.
    WARNING: This is not a standard signal generation technique for NATR.
    """
    natr = ta.NATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    natr_ma = ta.SMA(natr, timeperiod=ma_period)

    data["NATR_Signal"] = _generate_signals(
        condition_buy=natr > natr_ma,  # Volatility increasing / above average
        condition_sell=natr < natr_ma,  # Volatility decreasing / below average
    )
    # data['NATR'] = natr
    # data['NATR_MA'] = natr_ma
    print(
        "Warning: NATR_indicator revised logic (NATR vs MA) is unconventional for Buy/Sell signals."
    )
    return data


def TRANGE_indicator_v2(data, ma_period=14):
    """
    Vectorized True Range (TRANGE) indicator signals.
    Revised logic: Compare TRANGE to its moving average.
    WARNING: This is not a standard signal generation technique for TRANGE.
    """
    trange = ta.TRANGE(data["High"], data["Low"], data["Close"])
    trange_ma = ta.SMA(trange, timeperiod=ma_period)

    data["TRANGE_Signal"] = _generate_signals(
        condition_buy=trange > trange_ma,  # Volatility increasing / above average
        condition_sell=trange < trange_ma,  # Volatility decreasing / below average
    )
    # data['TRANGE'] = trange
    # data['TRANGE_MA'] = trange_ma
    print(
        "Warning: TRANGE_indicator revised logic (TRANGE vs MA) is unconventional for Buy/Sell signals."
    )
    return data


# --- Volatility Indicators ---


def ATR_indicator(data, timeperiod=14):
    """Vectorized Average True Range (ATR) indicator signals."""
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Original logic: > 20 Buy (High Volatility?), < 10 Sell (Low Volatility?) - Uncommon use
    data["ATR_Signal"] = _generate_signals(
        condition_buy=atr > 20, condition_sell=atr < 10
    )
    # data['ATR'] = atr
    print(
        "Warning: Using fixed ATR levels (10, 20) for Buy/Sell signals in ATR_indicator is unconventional."
    )
    return data


def NATR_indicator(data, timeperiod=14):
    """Vectorized Normalized Average True Range (NATR) indicator signals."""
    natr = ta.NATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    # Original logic: > 20 Buy, < 10 Sell - Uncommon use (NATR is percentage)
    data["NATR_Signal"] = _generate_signals(
        condition_buy=natr > 20, condition_sell=natr < 10
    )
    # data['NATR'] = natr
    print(
        "Warning: Using fixed NATR levels (10, 20) for Buy/Sell signals in NATR_indicator is unconventional."
    )
    return data


def TRANGE_indicator(data):
    """Vectorized True Range (TRANGE) indicator signals."""
    trange = ta.TRANGE(data["High"], data["Low"], data["Close"])
    # Original logic: > 20 Buy, < 10 Sell - Uncommon use
    data["TRANGE_Signal"] = _generate_signals(
        condition_buy=trange > 20, condition_sell=trange < 10
    )
    # data['TRANGE'] = trange
    print(
        "Warning: Using fixed TRANGE levels (10, 20) for Buy/Sell signals in TRANGE_indicator is unconventional."
    )
    return data


# --- Pattern Recognition ---


def _pattern_signals(pattern_series):
    """Helper for standard pattern recognition signals."""
    return _generate_signals(
        condition_buy=pattern_series > 0,  # Bullish pattern (e.g., 100)
        condition_sell=pattern_series < 0,  # Bearish pattern (e.g., -100)
    )


# Define ALL CDL functions using the pattern
def CDL2CROWS_indicator(data):
    """Vectorized Two Crows (CDL2CROWS) indicator signals."""
    pattern = ta.CDL2CROWS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL2CROWS_Signal"] = _pattern_signals(pattern)
    return data


def CDL3BLACKCROWS_indicator(data):
    """Vectorized Three Black Crows (CDL3BLACKCROWS) indicator signals."""
    pattern = ta.CDL3BLACKCROWS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3BLACKCROWS_Signal"] = _pattern_signals(pattern)
    return data


def CDL3INSIDE_indicator(data):
    """Vectorized Three Inside Up/Down (CDL3INSIDE) indicator signals."""
    pattern = ta.CDL3INSIDE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3INSIDE_Signal"] = _pattern_signals(pattern)
    return data


def CDL3LINESTRIKE_indicator(data):
    """Vectorized Three-Line Strike (CDL3LINESTRIKE) indicator signals."""
    pattern = ta.CDL3LINESTRIKE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3LINESTRIKE_Signal"] = _pattern_signals(pattern)
    return data


def CDL3OUTSIDE_indicator(data):
    """Vectorized Three Outside Up/Down (CDL3OUTSIDE) indicator signals."""
    pattern = ta.CDL3OUTSIDE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDL3OUTSIDE_Signal"] = _pattern_signals(pattern)
    return data


def CDL3STARSINSOUTH_indicator(data):
    """Vectorized Three Stars In The South (CDL3STARSINSOUTH) indicator signals."""
    pattern = ta.CDL3STARSINSOUTH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDL3STARSINSOUTH_Signal"] = _pattern_signals(pattern)
    return data


def CDL3WHITESOLDIERS_indicator(data):
    """Vectorized Three Advancing White Soldiers (CDL3WHITESOLDIERS) indicator signals."""
    pattern = ta.CDL3WHITESOLDIERS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDL3WHITESOLDIERS_Signal"] = _pattern_signals(pattern)
    return data


def CDLABANDONEDBABY_indicator(data, penetration=0):
    """Vectorized Abandoned Baby (CDLABANDONEDBABY) indicator signals."""
    pattern = ta.CDLABANDONEDBABY(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLABANDONEDBABY_Signal"] = _pattern_signals(pattern)
    return data


def CDLADVANCEBLOCK_indicator(data):
    """Vectorized Advance Block (CDLADVANCEBLOCK) indicator signals."""
    pattern = ta.CDLADVANCEBLOCK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLADVANCEBLOCK_Signal"] = _pattern_signals(pattern)
    return data


def CDLBELTHOLD_indicator(data):
    """Vectorized Belt-hold (CDLBELTHOLD) indicator signals."""
    pattern = ta.CDLBELTHOLD(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLBELTHOLD_Signal"] = _pattern_signals(pattern)
    return data


def CDLBREAKAWAY_indicator(data):
    """Vectorized Breakaway (CDLBREAKAWAY) indicator signals."""
    pattern = ta.CDLBREAKAWAY(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLBREAKAWAY_Signal"] = _pattern_signals(pattern)
    return data


def CDLCLOSINGMARUBOZU_indicator(data):
    """Vectorized Closing Marubozu (CDLCLOSINGMARUBOZU) indicator signals."""
    pattern = ta.CDLCLOSINGMARUBOZU(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCLOSINGMARUBOZU_Signal"] = _pattern_signals(pattern)
    return data


def CDLCONCEALBABYSWALL_indicator(data):
    """Vectorized Concealing Baby Swallow (CDLCONCEALBABYSWALL) indicator signals."""
    pattern = ta.CDLCONCEALBABYSWALL(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCONCEALBABYSWALL_Signal"] = _pattern_signals(pattern)
    return data


def CDLCOUNTERATTACK_indicator(data):
    """Vectorized Counterattack (CDLCOUNTERATTACK) indicator signals."""
    pattern = ta.CDLCOUNTERATTACK(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLCOUNTERATTACK_Signal"] = _pattern_signals(pattern)
    return data


def CDLDARKCLOUDCOVER_indicator(data, penetration=0):
    """Vectorized Dark Cloud Cover (CDLDARKCLOUDCOVER) indicator signals."""
    pattern = ta.CDLDARKCLOUDCOVER(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLDARKCLOUDCOVER_Signal"] = _pattern_signals(pattern)
    return data


def CDLDOJI_indicator(data):
    """Vectorized Doji (CDLDOJI) indicator signals."""
    pattern = ta.CDLDOJI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLDOJI_Signal"] = _pattern_signals(pattern)
    return data


def CDLDOJISTAR_indicator(data):
    """Vectorized Doji Star (CDLDOJISTAR) indicator signals."""
    pattern = ta.CDLDOJISTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLDRAGONFLYDOJI_indicator(data):
    """Vectorized Dragonfly Doji (CDLDRAGONFLYDOJI) indicator signals."""
    pattern = ta.CDLDRAGONFLYDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLDRAGONFLYDOJI_Signal"] = _pattern_signals(pattern)
    return data


def CDLENGULFING_indicator(data):
    """Vectorized Engulfing Pattern (CDLENGULFING) indicator signals."""
    pattern = ta.CDLENGULFING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLENGULFING_Signal"] = _pattern_signals(pattern)
    return data


def CDLEVENINGDOJISTAR_indicator(data, penetration=0):
    """Vectorized Evening Doji Star (CDLEVENINGDOJISTAR) indicator signals."""
    pattern = ta.CDLEVENINGDOJISTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLEVENINGDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLEVENINGSTAR_indicator(data, penetration=0):
    """Vectorized Evening Star (CDLEVENINGSTAR) indicator signals."""
    pattern = ta.CDLEVENINGSTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLEVENINGSTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLGAPSIDESIDEWHITE_indicator(data):
    """Vectorized Up/Down-gap side-by-side white lines (CDLGAPSIDESIDEWHITE) indicator signals."""
    pattern = ta.CDLGAPSIDESIDEWHITE(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLGAPSIDESIDEWHITE_Signal"] = _pattern_signals(pattern)
    return data


def CDLGRAVESTONEDOJI_indicator(data):
    """Vectorized Gravestone Doji (CDLGRAVESTONEDOJI) indicator signals."""
    pattern = ta.CDLGRAVESTONEDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLGRAVESTONEDOJI_Signal"] = _pattern_signals(pattern)
    return data


def CDLHAMMER_indicator(data):
    """Vectorized Hammer (CDLHAMMER) indicator signals."""
    pattern = ta.CDLHAMMER(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHAMMER_Signal"] = _pattern_signals(pattern)
    return data


def CDLHANGINGMAN_indicator(data):
    """Vectorized Hanging Man (CDLHANGINGMAN) indicator signals."""
    pattern = ta.CDLHANGINGMAN(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHANGINGMAN_Signal"] = _pattern_signals(pattern)
    return data


def CDLHARAMI_indicator(data):
    """Vectorized Harami Pattern (CDLHARAMI) indicator signals."""
    pattern = ta.CDLHARAMI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHARAMI_Signal"] = _pattern_signals(pattern)
    return data


def CDLHARAMICROSS_indicator(data):
    """Vectorized Harami Cross Pattern (CDLHARAMICROSS) indicator signals."""
    pattern = ta.CDLHARAMICROSS(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHARAMICROSS_Signal"] = _pattern_signals(pattern)
    return data


def CDLHIGHWAVE_indicator(data):
    """Vectorized High-Wave Candle (CDLHIGHWAVE) indicator signals."""
    pattern = ta.CDLHIGHWAVE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIGHWAVE_Signal"] = _pattern_signals(pattern)
    return data


def CDLHIKKAKE_indicator(data):
    """Vectorized Hikkake Pattern (CDLHIKKAKE) indicator signals."""
    pattern = ta.CDLHIKKAKE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIKKAKE_Signal"] = _pattern_signals(pattern)
    return data


def CDLHIKKAKEMOD_indicator(data):
    """Vectorized Modified Hikkake Pattern (CDLHIKKAKEMOD) indicator signals."""
    pattern = ta.CDLHIKKAKEMOD(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHIKKAKEMOD_Signal"] = _pattern_signals(pattern)
    return data


def CDLHOMINGPIGEON_indicator(data):
    """Vectorized Homing Pigeon (CDLHOMINGPIGEON) indicator signals."""
    pattern = ta.CDLHOMINGPIGEON(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLHOMINGPIGEON_Signal"] = _pattern_signals(pattern)
    return data


def CDLIDENTICAL3CROWS_indicator(data):
    """Vectorized Identical Three Crows (CDLIDENTICAL3CROWS) indicator signals."""
    pattern = ta.CDLIDENTICAL3CROWS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLIDENTICAL3CROWS_Signal"] = _pattern_signals(pattern)
    return data


def CDLINNECK_indicator(data):
    """Vectorized In-Neck Pattern (CDLINNECK) indicator signals."""
    pattern = ta.CDLINNECK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLINNECK_Signal"] = _pattern_signals(pattern)
    return data


def CDLINVERTEDHAMMER_indicator(data):
    """Vectorized Inverted Hammer (CDLINVERTEDHAMMER) indicator signals."""
    pattern = ta.CDLINVERTEDHAMMER(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLINVERTEDHAMMER_Signal"] = _pattern_signals(pattern)
    return data


def CDLKICKING_indicator(data):
    """Vectorized Kicking (CDLKICKING) indicator signals."""
    pattern = ta.CDLKICKING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLKICKING_Signal"] = _pattern_signals(pattern)
    return data


def CDLKICKINGBYLENGTH_indicator(data):
    """Vectorized Kicking - bull/bear determined by the longer marubozu (CDLKICKINGBYLENGTH) indicator signals."""
    pattern = ta.CDLKICKINGBYLENGTH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLKICKINGBYLENGTH_Signal"] = _pattern_signals(pattern)
    return data


def CDLLADDERBOTTOM_indicator(data):
    """Vectorized Ladder Bottom (CDLLADDERBOTTOM) indicator signals."""
    pattern = ta.CDLLADDERBOTTOM(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLLADDERBOTTOM_Signal"] = _pattern_signals(pattern)
    return data


def CDLLONGLEGGEDDOJI_indicator(data):
    """Vectorized Long Legged Doji (CDLLONGLEGGEDDOJI) indicator signals."""
    pattern = ta.CDLLONGLEGGEDDOJI(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLLONGLEGGEDDOJI_Signal"] = _pattern_signals(pattern)
    return data


def CDLLONGLINE_indicator(data):
    """Vectorized Long Line Candle (CDLLONGLINE) indicator signals."""
    pattern = ta.CDLLONGLINE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLLONGLINE_Signal"] = _pattern_signals(pattern)
    return data


def CDLMARUBOZU_indicator(data):
    """Vectorized Marubozu (CDLMARUBOZU) indicator signals."""
    pattern = ta.CDLMARUBOZU(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLMARUBOZU_Signal"] = _pattern_signals(pattern)
    return data


def CDLMATCHINGLOW_indicator(data):
    """Vectorized Matching Low (CDLMATCHINGLOW) indicator signals."""
    pattern = ta.CDLMATCHINGLOW(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLMATCHINGLOW_Signal"] = _pattern_signals(pattern)
    return data


def CDLMATHOLD_indicator(data, penetration=0):
    """Vectorized Mat Hold (CDLMATHOLD) indicator signals."""
    pattern = ta.CDLMATHOLD(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMATHOLD_Signal"] = _pattern_signals(pattern)
    return data


def CDLMORNINGDOJISTAR_indicator(data, penetration=0):
    """Vectorized Morning Doji Star (CDLMORNINGDOJISTAR) indicator signals."""
    pattern = ta.CDLMORNINGDOJISTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMORNINGDOJISTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLMORNINGSTAR_indicator(data, penetration=0):
    """Vectorized Morning Star (CDLMORNINGSTAR) indicator signals."""
    pattern = ta.CDLMORNINGSTAR(
        data["Open"], data["High"], data["Low"], data["Close"], penetration=penetration
    )
    data["CDLMORNINGSTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLONNECK_indicator(data):
    """Vectorized On-Neck Pattern (CDLONNECK) indicator signals."""
    pattern = ta.CDLONNECK(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLONNECK_Signal"] = _pattern_signals(pattern)
    return data


def CDLPIERCING_indicator(data):
    """Vectorized Piercing Pattern (CDLPIERCING) indicator signals."""
    pattern = ta.CDLPIERCING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLPIERCING_Signal"] = _pattern_signals(pattern)
    return data


def CDLRICKSHAWMAN_indicator(data):
    """Vectorized Rickshaw Man (CDLRICKSHAWMAN) indicator signals."""
    pattern = ta.CDLRICKSHAWMAN(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLRICKSHAWMAN_Signal"] = _pattern_signals(pattern)
    return data


def CDLRISEFALL3METHODS_indicator(data):
    """Vectorized Rising/Falling Three Methods (CDLRISEFALL3METHODS) indicator signals."""
    pattern = ta.CDLRISEFALL3METHODS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLRISEFALL3METHODS_Signal"] = _pattern_signals(pattern)
    return data


def CDLSEPARATINGLINES_indicator(data):
    """Vectorized Separating Lines (CDLSEPARATINGLINES) indicator signals."""
    pattern = ta.CDLSEPARATINGLINES(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSEPARATINGLINES_Signal"] = _pattern_signals(pattern)
    return data


def CDLSHOOTINGSTAR_indicator(data):
    """Vectorized Shooting Star (CDLSHOOTINGSTAR) indicator signals."""
    pattern = ta.CDLSHOOTINGSTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSHOOTINGSTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLSHORTLINE_indicator(data):
    """Vectorized Short Line Candle (CDLSHORTLINE) indicator signals."""
    pattern = ta.CDLSHORTLINE(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSHORTLINE_Signal"] = _pattern_signals(pattern)
    return data


def CDLSPINNINGTOP_indicator(data):
    """Vectorized Spinning Top (CDLSPINNINGTOP) indicator signals."""
    pattern = ta.CDLSPINNINGTOP(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLSPINNINGTOP_Signal"] = _pattern_signals(pattern)
    return data


def CDLSTALLEDPATTERN_indicator(data):
    """Vectorized Stalled Pattern (CDLSTALLEDPATTERN) indicator signals."""
    pattern = ta.CDLSTALLEDPATTERN(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSTALLEDPATTERN_Signal"] = _pattern_signals(pattern)
    return data


def CDLSTICKSANDWICH_indicator(data):
    """Vectorized Stick Sandwich (CDLSTICKSANDWICH) indicator signals."""
    pattern = ta.CDLSTICKSANDWICH(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLSTICKSANDWICH_Signal"] = _pattern_signals(pattern)
    return data


def CDLTAKURI_indicator(data):
    """Vectorized Takuri (Dragonfly Doji with very long lower shadow) (CDLTAKURI) indicator signals."""
    pattern = ta.CDLTAKURI(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTAKURI_Signal"] = _pattern_signals(pattern)
    return data


def CDLTASUKIGAP_indicator(data):
    """Vectorized Tasuki Gap (CDLTASUKIGAP) indicator signals."""
    pattern = ta.CDLTASUKIGAP(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTASUKIGAP_Signal"] = _pattern_signals(pattern)
    return data


def CDLTHRUSTING_indicator(data):
    """Vectorized Thrusting Pattern (CDLTHRUSTING) indicator signals."""
    pattern = ta.CDLTHRUSTING(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTHRUSTING_Signal"] = _pattern_signals(pattern)
    return data


def CDLTRISTAR_indicator(data):
    """Vectorized Tristar Pattern (CDLTRISTAR) indicator signals."""
    pattern = ta.CDLTRISTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLTRISTAR_Signal"] = _pattern_signals(pattern)
    return data


def CDLUNIQUE3RIVER_indicator(data):
    """Vectorized Unique 3 River (CDLUNIQUE3RIVER) indicator signals."""
    pattern = ta.CDLUNIQUE3RIVER(data["Open"], data["High"], data["Low"], data["Close"])
    data["CDLUNIQUE3RIVER_Signal"] = _pattern_signals(pattern)
    return data


def CDLUPSIDEGAP2CROWS_indicator(data):
    """Vectorized Upside Gap Two Crows (CDLUPSIDEGAP2CROWS) indicator signals."""
    pattern = ta.CDLUPSIDEGAP2CROWS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLUPSIDEGAP2CROWS_Signal"] = _pattern_signals(pattern)
    return data


def CDLXSIDEGAP3METHODS_indicator(data):
    """Vectorized Upside/Downside Gap Three Methods (CDLXSIDEGAP3METHODS) indicator signals."""
    pattern = ta.CDLXSIDEGAP3METHODS(
        data["Open"], data["High"], data["Low"], data["Close"]
    )
    data["CDLXSIDEGAP3METHODS_Signal"] = _pattern_signals(pattern)
    return data


# --- Revised Statistic Functions ---
# flawed usage


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
    # data['BETA'] = beta
    print(
        "Warning: BETA_indicator Buy/Sell signals based on Beta > 1 or < 1 are highly context-dependent and may not be meaningful."
    )
    return data


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
    # data['CORREL'] = correl
    print(
        "Warning: CORREL_indicator Buy/Sell signals based on fixed correlation levels are arbitrary and context-dependent."
    )
    return data


def LINEARREG_INTERCEPT_indicator_v2(data, timeperiod=14):
    """
    Vectorized Linear Regression Intercept (LINEARREG_INTERCEPT) indicator signals.
    Revised logic: Compares Close to the forecast value (LINEARREG), not the intercept.
    This makes it logically identical to LINEARREG_indicator.
    """
    linearreg = ta.LINEARREG(data["Close"], timeperiod=timeperiod)
    # linearreg_intercept = ta.LINEARREG_INTERCEPT(data["Close"], timeperiod=timeperiod) # Original calculation

    # Revised logic compares to the forecast value
    data["LINEARREG_INTERCEPT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg,
        condition_sell=data["Close"] < linearreg,
    )
    # data['LINEARREG_INTERCEPT_Orig'] = linearreg_intercept # If you need the original value
    # data['LINEARREG'] = linearreg
    print(
        "Note: LINEARREG_INTERCEPT_indicator logic revised to compare Close vs LINEARREG (forecast), making it equivalent to LINEARREG_indicator."
    )
    return data


def STDDEV_indicator_v2(data, timeperiod=20, nbdev=1, ma_period=20):
    """
    Vectorized Standard Deviation (STDDEV) indicator signals.
    Revised logic: Compare STDDEV to its moving average.
    WARNING: This is not a standard signal generation technique for STDDEV.
    """
    stddev = ta.STDDEV(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    stddev_ma = ta.SMA(stddev, timeperiod=ma_period)

    data["STDDEV_Signal"] = _generate_signals(
        condition_buy=stddev > stddev_ma,  # Volatility increasing / above average
        condition_sell=stddev < stddev_ma,  # Volatility decreasing / below average
    )
    # data['STDDEV'] = stddev
    # data['STDDEV_MA'] = stddev_ma
    print(
        "Warning: STDDEV_indicator revised logic (STDDEV vs MA) is unconventional for Buy/Sell signals."
    )
    return data


def VAR_indicator_v2(data, timeperiod=5, nbdev=1, ma_period=5):
    """
    Vectorized Variance (VAR) indicator signals.
    Revised logic: Compare VAR to its moving average.
    WARNING: This is not a standard signal generation technique for VAR.
    """
    var = ta.VAR(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    var_ma = ta.SMA(var, timeperiod=ma_period)

    data["VAR_Signal"] = _generate_signals(
        condition_buy=var > var_ma,  # Volatility increasing / above average
        condition_sell=var < var_ma,  # Volatility decreasing / below average
    )
    # data['VAR'] = var
    # data['VAR_MA'] = var_ma
    print(
        "Warning: VAR_indicator revised logic (VAR vs MA) is unconventional for Buy/Sell signals."
    )
    return data


# --- Statistic Functions ---


def BETA_indicator(data, timeperiod=5):
    """Vectorized Beta (BETA) indicator signals."""
    beta = ta.BETA(data["High"], data["Low"], timeperiod=timeperiod)
    # Original logic: > 1 Buy, < 1 Sell. Interpretation might depend on context.
    data["BETA_Signal"] = _generate_signals(
        condition_buy=beta > 1, condition_sell=beta < 1
    )
    # data['BETA'] = beta
    return data


def CORREL_indicator(data, timeperiod=30):
    """Vectorized Pearson's Correlation Coefficient (CORREL) indicator signals."""
    correl = ta.CORREL(data["High"], data["Low"], timeperiod=timeperiod)
    # Original logic: > 0.5 Buy (Strong positive corr), < -0.5 Sell (Strong negative corr)
    data["CORREL_Signal"] = _generate_signals(
        condition_buy=correl > 0.5, condition_sell=correl < -0.5
    )
    # data['CORREL'] = correl
    return data


# superseeded
def LINEARREG_indicator(data, timeperiod=14):
    """Vectorized Linear Regression (LINEARREG) indicator signals."""
    linearreg = ta.LINEARREG(data["Close"], timeperiod=timeperiod)
    # Signal based on price crossing the linear regression line
    data["LINEARREG_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg,
        condition_sell=data["Close"] < linearreg,
    )
    # data['LINEARREG'] = linearreg
    return data


def LINEARREG_ANGLE_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Angle (LINEARREG_ANGLE) indicator signals."""
    linearreg_angle = ta.LINEARREG_ANGLE(data["Close"], timeperiod=timeperiod)
    # Signal based on the angle (trend direction)
    data["LINEARREG_ANGLE_Signal"] = _generate_signals(
        condition_buy=linearreg_angle > 0,  # Upward angle
        condition_sell=linearreg_angle < 0,  # Downward angle
    )
    # data['LINEARREG_ANGLE'] = linearreg_angle
    return data


# superseeded
def LINEARREG_INTERCEPT_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Intercept (LINEARREG_INTERCEPT) indicator signals."""
    linearreg_intercept = ta.LINEARREG_INTERCEPT(data["Close"], timeperiod=timeperiod)
    # Original logic: Close > Intercept Buy, Close < Intercept Sell.
    data["LINEARREG_INTERCEPT_Signal"] = _generate_signals(
        condition_buy=data["Close"] > linearreg_intercept,
        condition_sell=data["Close"] < linearreg_intercept,
    )
    # data['LINEARREG_INTERCEPT'] = linearreg_intercept
    print(
        "Warning: Comparing Close to LINEARREG_INTERCEPT for signals in LINEARREG_INTERCEPT_indicator might be unconventional."
    )
    return data


def LINEARREG_SLOPE_indicator(data, timeperiod=14):
    """Vectorized Linear Regression Slope (LINEARREG_SLOPE) indicator signals."""
    linearreg_slope = ta.LINEARREG_SLOPE(data["Close"], timeperiod=timeperiod)
    # Signal based on the slope (trend strength/direction)
    data["LINEARREG_SLOPE_Signal"] = _generate_signals(
        condition_buy=linearreg_slope > 0,  # Positive slope
        condition_sell=linearreg_slope < 0,  # Negative slope
    )
    # data['LINEARREG_SLOPE'] = linearreg_slope
    return data


def STDDEV_indicator(data, timeperiod=20, nbdev=1):
    """Vectorized Standard Deviation (STDDEV) indicator signals."""
    stddev = ta.STDDEV(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    # Original logic: > 20 Buy, < 10 Sell. Using Std Dev value directly is uncommon for signals.
    data["STDDEV_Signal"] = _generate_signals(
        condition_buy=stddev > 20, condition_sell=stddev < 10
    )
    # data['STDDEV'] = stddev
    print(
        "Warning: Using fixed STDDEV levels (10, 20) for Buy/Sell signals in STDDEV_indicator is unconventional."
    )
    return data


def TSF_indicator(data, timeperiod=14):
    """Vectorized Time Series Forecast (TSF) indicator signals."""
    tsf = ta.TSF(data["Close"], timeperiod=timeperiod)
    # Signal based on price crossing the forecast line
    data["TSF_Signal"] = _generate_signals(
        condition_buy=data["Close"] > tsf, condition_sell=data["Close"] < tsf
    )
    # data['TSF'] = tsf
    return data


def VAR_indicator(data, timeperiod=5, nbdev=1):
    """Vectorized Variance (VAR) indicator signals."""
    var = ta.VAR(data["Close"], timeperiod=timeperiod, nbdev=nbdev)
    # Original logic: > 20 Buy, < 10 Sell. Using Variance value directly is uncommon for signals.
    data["VAR_Signal"] = _generate_signals(
        condition_buy=var > 20, condition_sell=var < 10
    )
    # data['VAR'] = var
    print(
        "Warning: Using fixed VAR levels (10, 20) for Buy/Sell signals in VAR_indicator is unconventional."
    )
    return data


"""
NEW Indicators

Explanation and Notes:

    Ichimoku Cloud (ichimoku_cloud_indicator):

        Calculates all five standard components (Tenkan, Kijun, Senkou A, Senkou B, Chikou).

        Uses Pandas .rolling() with .max() and .min() for the highest high / lowest low calculations.

        Uses Pandas .shift() to correctly plot Senkou spans ahead (shift(period_kijun)) and Chikou span behind (shift(-period_kijun)). This means there will be NaNs at the beginning/end of these shifted columns.

        Includes a basic signal: Buy if price is above the cloud (above both Senkou A and B), Sell if below. Real Ichimoku trading involves more signal combinations.

    Keltner Channels (keltner_channels_indicator):

        Uses ta.EMA for the middle line and ta.ATR for the range calculation.

        Combines these using standard arithmetic for the upper and lower bands based on the multiplier.

        Includes a basic breakout signal: Buy on close above upper band, Sell on close below lower band.

    VWAP (vwap_indicator):

        Calculates the rolling VWAP over the specified window.

        Uses Typical Price (H+L+C)/3.

        Uses Pandas .rolling().sum() to get the necessary sums for the calculation.

        Includes a check for the 'Volume' column.

        Handles potential division by zero if the rolling volume sum is zero (by replacing with NaN and then forward-filling).

        Adds a basic signal comparing Close to the calculated VWAP.

        Important: Explicitly notes that this is not the daily resetting VWAP commonly used in intraday analysis. A separate function using groupby(date) and cumsum would be needed for that specific logic.




ichimoku_cloud_indicator:

Category: overlap_studies

Reasoning: Ichimoku Cloud components (Tenkan-sen, Kijun-sen, Senkou Spans) are plotted directly on the price chart to provide potential support/resistance levels, trend direction, and signal crossovers, similar to moving averages and Bollinger Bands.

keltner_channels_indicator:

Category: overlap_studies

Reasoning: Like Bollinger Bands, Keltner Channels are plotted as bands around the price action, defining expected price movement ranges. Although they use ATR (a volatility indicator) in their calculation, their primary use and representation place them in Overlap Studies.

vwap_indicator:

Category: volume_indicators

Reasoning: The defining characteristic of VWAP is its explicit incorporation of Volume data to weight the average price. While it produces a price level often plotted on the chart, its calculation is fundamentally driven by volume, making volume_indicators the most accurate category.        
"""


# --- New Indicator Functions ---


# overlap_studies
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
    # Input check
    required_cols = ["High", "Low", "Close"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")

    high_prices = data["High"]
    low_prices = data["Low"]
    close_prices = data["Close"]

    # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for the past 9 periods.
    nine_period_high = high_prices.rolling(window=period_tenkan).max()
    nine_period_low = low_prices.rolling(window=period_tenkan).min()
    data["Ichi_Tenkan"] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for the past 26 periods.
    twenty_six_period_high = high_prices.rolling(window=period_kijun).max()
    twenty_six_period_low = low_prices.rolling(window=period_kijun).min()
    data["Ichi_Kijun"] = (twenty_six_period_high + twenty_six_period_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead.
    data["Ichi_SenkouA"] = ((data["Ichi_Tenkan"] + data["Ichi_Kijun"]) / 2).shift(
        period_kijun
    )

    # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for the past 52 periods, plotted 26 periods ahead.
    fifty_two_period_high = high_prices.rolling(window=period_senkou_b).max()
    fifty_two_period_low = low_prices.rolling(window=period_senkou_b).min()
    data["Ichi_SenkouB"] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(
        period_kijun
    )

    # Chikou Span (Lagging Span): Current Closing Price plotted 26 periods behind.
    data["Ichi_Chikou"] = close_prices.shift(-period_kijun)

    # --- Basic Signal Generation (Price vs Cloud) ---
    # Buy if Close is above both Senkou Spans (above the cloud)
    # Sell if Close is below both Senkou Spans (below the cloud)
    # Note: Senkou A and B can cross, so check both.
    above_cloud = (close_prices > data["Ichi_SenkouA"]) & (
        close_prices > data["Ichi_SenkouB"]
    )
    below_cloud = (close_prices < data["Ichi_SenkouA"]) & (
        close_prices < data["Ichi_SenkouB"]
    )

    data["Ichimoku_Signal"] = _generate_signals(
        condition_buy=above_cloud, condition_sell=below_cloud
    )
    # Other common signals (not implemented here):
    # - Tenkan/Kijun cross
    # - Price vs Kijun cross
    # - Chikou Span vs Price cross (from 26 periods ago)

    return data


# overlap_studies
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
    # Input check
    required_cols = ["High", "Low", "Close"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")

    # Calculate EMA for the middle line
    data["KC_Middle"] = ta.EMA(data["Close"], timeperiod=period_ema)

    # Calculate ATR
    atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=period_atr)

    # Calculate Upper and Lower Bands
    data["KC_Upper"] = data["KC_Middle"] + (multiplier * atr)
    data["KC_Lower"] = data["KC_Middle"] - (multiplier * atr)

    # --- Basic Signal Generation (Breakout) ---
    # Buy if Close breaks above the Upper Channel
    # Sell if Close breaks below the Lower Channel
    data["Keltner_Signal"] = _generate_signals(
        condition_buy=data["Close"] > data["KC_Upper"],
        condition_sell=data["Close"] < data["KC_Lower"],
    )

    return data


# volume_indicators
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
    # Input check
    required_cols = ["High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must include columns: {required_cols}")
    if data["Volume"].isnull().any() or (data["Volume"] < 0).any():
        print(
            "Warning: VWAP calculation encountered missing or negative Volume data. Results may be inaccurate."
        )

    # Calculate Typical Price
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3

    # Calculate rolling sum of (Typical Price * Volume) and Volume
    tp_vol = typical_price * data["Volume"]
    sum_tp_vol = tp_vol.rolling(window=window, min_periods=window).sum()
    sum_vol = data["Volume"].rolling(window=window, min_periods=window).sum()

    # Calculate VWAP, handle potential division by zero
    # Replace 0 volume sum with NaN to avoid division error, then fill NaNs
    sum_vol_safe = sum_vol.replace(0, np.nan)
    data["VWAP"] = sum_tp_vol / sum_vol_safe
    data["VWAP"] = data["VWAP"].fillna(method="ffill")  # Fill initial NaNs

    # --- Basic Signal Generation (Price vs VWAP) ---
    # Buy if Close is above VWAP
    # Sell if Close is below VWAP
    data["VWAP_Signal"] = _generate_signals(
        condition_buy=data["Close"] > data["VWAP"],
        condition_sell=data["Close"] < data["VWAP"],
    )

    return data


# --- Example Usage ---
if __name__ == "__main__":
    # Create sample data (longer for Ichimoku shifts)
    num_rows = 200
    data = {
        "Open": np.random.rand(num_rows) * 10 + 100,
        "High": np.random.rand(num_rows) * 5 + 105,
        "Low": 100 - np.random.rand(num_rows) * 5,
        "Close": np.random.rand(num_rows) * 10 + 100,
        "Volume": np.random.rand(num_rows) * 10000 + 50000,
    }
    # Create a DatetimeIndex
    index = pd.date_range(
        start="2023-01-01", periods=num_rows, freq="B"
    )  # Business days
    df = pd.DataFrame(data, index=index)

    # Ensure High is >= Open/Close and Low is <= Open/Close
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    # Add some trend
    trend = np.sin(np.linspace(0, 15, len(df))) * 5 + np.linspace(0, 20, len(df))
    df["Close"] = df["Close"] * 0.3 + (100 + trend) * 0.7
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * (
        1 + np.random.randn(len(df)) * 0.005
    )
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.rand(len(df)) * 2
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.rand(len(df)) * 2
    df["Volume"] = (
        df["Volume"] * (1 + df["Close"].pct_change().fillna(0).abs() * 2)
    ).abs()  # Make volume somewhat related to price change

    print("Original DataFrame tail:")
    print(df.tail())

    # Apply the new indicators
    df_ichi = ichimoku_cloud_indicator(df.copy())
    df_kc = keltner_channels_indicator(df.copy())
    df_vwap = vwap_indicator(df.copy(), window=20)  # Using 20 period rolling VWAP

    print("\n--- Ichimoku Cloud Results (Tail) ---")
    ichi_cols = [
        "Close",
        "Ichi_Tenkan",
        "Ichi_Kijun",
        "Ichi_SenkouA",
        "Ichi_SenkouB",
        "Ichi_Chikou",
        "Ichimoku_Signal",
    ]
    print(df_ichi[ichi_cols].tail(10))
    # Note how SenkouA/B and Chikou have NaNs at the end/beginning due to shifting

    print("\n--- Keltner Channels Results (Tail) ---")
    kc_cols = ["Close", "KC_Lower", "KC_Middle", "KC_Upper", "Keltner_Signal"]
    print(df_kc[kc_cols].tail(10))

    print("\n--- Rolling VWAP Results (Tail) ---")
    vwap_cols = ["Close", "Volume", "VWAP", "VWAP_Signal"]
    print(df_vwap[vwap_cols].tail(10))

# --- Example Usage ---
if __name__ == "__main__":
    # Create sample data
    data = {
        "Open": np.random.rand(100) * 10 + 100,
        "High": np.random.rand(100) * 5 + 105,
        "Low": 100 - np.random.rand(100) * 5,
        "Close": np.random.rand(100) * 10 + 100,
        "Volume": np.random.rand(100) * 10000 + 5000,
    }
    df = pd.DataFrame(data)
    # Ensure High is >= Open/Close and Low is <= Open/Close
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    # Add some trend for MACD example
    df["Close"] = df["Close"] + np.linspace(0, 15, len(df))

    # Apply a vectorized function
    df = BBANDS_indicator(
        df.copy()
    )  # Use copy to avoid modifying original df if needed
    df = MACD_indicator(df.copy())
    df = RSI_indicator(df.copy())
    df = CDLHAMMER_indicator(df.copy())  # Example pattern

    print("DataFrame with vectorized signals:")
    # Display relevant columns
    signal_cols = [col for col in df.columns if "_Signal" in col]
    print(df[["Close"] + signal_cols].tail(10))

    # You can check the signals for the last row
    print("\nLast row signals:")
    print(df.iloc[-1][signal_cols])

    # Example with MAVP
    df_mavp = df.copy()
    # Create the 'periods' column needed by the MAVP_indicator function
    # Example: Variable period based on volatility (ATR) - just an illustration
    atr_temp = ta.ATR(df_mavp["High"], df_mavp["Low"], df_mavp["Close"], timeperiod=10)
    df_mavp["periods"] = pd.Series(
        np.where(atr_temp > atr_temp.median(), 10.0, 30.0), index=df_mavp.index
    )
    df_mavp["periods"] = df_mavp["periods"].fillna(30.0)  # Fill initial NaNs

    # Now call MAVP_indicator
    try:
        df_mavp = MAVP_indicator(df_mavp)
        print("\nMAVP Example:")
        print(df_mavp[["Close", "periods", "MAVP_Signal"]].tail())
    except Exception as e:
        print(f"\nError calculating MAVP example: {e}")
