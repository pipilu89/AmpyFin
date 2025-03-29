import pandas as pd
import talib as ta
import numpy as np


# --- Helper Function for Common Logic ---
def _generate_signals(condition_buy, condition_sell, default="Hold"):
    """Uses np.select to generate signals based on boolean conditions."""
    conditions = [condition_buy, condition_sell]
    choices = ["Buy", "Sell"]
    return np.select(conditions, choices, default=default)


# --- (Keep unchanged functions from the previous version here) ---
# BBANDS_indicator, DEMA_indicator, EMA_indicator, HT_TRENDLINE_indicator,
# KAMA_indicator, MA_indicator, MAMA_indicator, MAVP_indicator,
# MIDPOINT_indicator, MIDPRICE_indicator, SAR_indicator, SAREXT_indicator,
# SMA_indicator, T3_indicator, TEMA_indicator, TRIMA_indicator, WMA_indicator
# APO_indicator, AROON_indicator, AROONOSC_indicator, BOP_indicator
# MACD_indicator, MACDEXT_indicator, MACDFIX_indicator
# MFI_indicator (already correct)
# MOM_indicator
# PPO_indicator, ROC_indicator, ROCP_indicator, ROCR_indicator, ROCR100_indicator
# RSI_indicator (already correct)
# STOCH_indicator, STOCHF_indicator, STOCHRSI_indicator (already correct)
# TRIX_indicator
# ULTOSC_indicator (already correct)
# WILLR_indicator (already correct)
# HT_DCPERIOD_indicator, HT_DCPHASE_indicator, HT_PHASOR_indicator, HT_SINE_indicator
# AVGPRICE_indicator, MEDPRICE_indicator, TYPPRICE_indicator, WCLPRICE_indicator
# CDL* indicators (logic based on pattern output is standard)
# LINEARREG_indicator, LINEARREG_ANGLE_indicator, LINEARREG_SLOPE_indicator
# TSF_indicator

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


# def PLUS_DI_indicator(data, timeperiod=14):
#     """
#     Vectorized Plus Directional Indicator (PLUS_DI) signals.
#     Revised Logic: Buy if DI+ is dominant (DI+ > DI-).
#     """
#     plus_di = ta.PLUS_DI(
#         data["High"], data["Low"], data["Close"], timeperiod=timeperiod
#     )
#     minus_di = ta.MINUS_DI(
#         data["High"], data["Low"], data["Close"], timeperiod=timeperiod
#     )

#     data["PLUS_DI_Signal"] = _generate_signals(
#         condition_buy=plus_di > minus_di,  # DI+ is dominant
#         condition_sell=minus_di > plus_di,  # DI- is dominant
#     )
#     # data['PLUS_DI'] = plus_di
#     # data['MINUS_DI'] = minus_di
#     return data


# Remove MINUS_DM_indicator and PLUS_DM_indicator as standalone signals

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


# --- Example Usage (Illustrating some revised functions) ---
if __name__ == "__main__":
    # Create sample data
    data = {
        "Open": np.random.rand(150) * 10 + 100,
        "High": np.random.rand(150) * 5 + 105,
        "Low": 100 - np.random.rand(150) * 5,
        "Close": np.random.rand(150) * 10 + 100,
        "Volume": np.random.rand(150) * 10000 + 50000,
    }
    df = pd.DataFrame(data)
    # Ensure High is >= Open/Close and Low is <= Open/Close
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    # Add some trend
    trend = np.sin(np.linspace(0, 10, len(df))) * 5 + np.linspace(0, 15, len(df))
    df["Close"] = df["Close"] * 0.3 + (100 + trend) * 0.7
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * (
        1 + np.random.randn(len(df)) * 0.005
    )
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.rand(len(df)) * 2
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.rand(len(df)) * 2

    # Apply some original and revised functions
    df_orig = df.copy()
    df_revised = df.copy()

    print("\n--- Applying ADX ---")
    df_revised = ADX_indicator_v2(df_revised)

    print("\n--- Applying CCI ---")
    df_revised = CCI_indicator_v2(df_revised)

    print("\n--- Applying AD (Volume) ---")
    df_revised = AD_indicator_v2(df_revised)

    print("\n--- Applying HT_TRENDMODE ---")
    df_revised = HT_TRENDMODE_indicator_v2(df_revised)

    print("\n--- Applying ATR (Volatility) ---")
    df_revised = ATR_indicator_v2(df_revised)

    print("\n--- Applying LINEARREG_INTERCEPT ---")
    df_revised = LINEARREG_INTERCEPT_indicator_v2(df_revised)
    # df_revised = LINEARREG_indicator(df_revised)  # For comparison

    print("\nDataFrame with REVISED signals (Tail):")
    signal_cols = [col for col in df_revised.columns if "_Signal" in col]
    print(df_revised[["Close"] + signal_cols].tail(15))

    # Compare LINREG signals
    # print(
    #     "\nComparison of LINREG and LINREG_INTERCEPT signals (should be identical with revised logic):"
    # )
    # print(
    #     df_revised[["Close", "LINEARREG_Signal", "LINEARREG_INTERCEPT_Signal"]].tail(10)
    # )
