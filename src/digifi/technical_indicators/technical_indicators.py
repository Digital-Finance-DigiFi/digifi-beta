import numpy as np
import pandas as pd
from src.digifi.utilities.general_utils import (verify_array, compare_array_len)



def sma(price_array: np.ndarray, period: int=15) -> pd.DataFrame:
    """
    Takes in an array of prices and returns a dataframe of SMA readings.
    Simple Moving Average (SMA) describes the direction of the trend, and is computed using the mean over the certain window of readings.
    """
    verify_array(array=price_array)
    return pd.Series(price_array).rolling(window=int(period)).mean().to_frame(name="{} SMA".format(period))



def ema(price_array: np.ndarray, period: int=20, smoothing: int=2) -> pd.DataFrame:
    """
    Takes in an array of prices and returns a dataframe of EMA readings.
    Exponential Moving Average (EMA) describes the direction of the trend, and requires previous EMA and the latest price to compute;
    the first EMA reading will be same as SMA.
    """
    verify_array(array=price_array)
    period = int(period)
    smoothing = int(smoothing)
    multiplier = smoothing/(1+period)
    s = pd.Series(price_array)
    ema = np.nan*np.ones(len(price_array))
    ema[period-1] = s.iloc[0:period].mean()
    for i in range(period, len(s)):
        ema[i] = s.iloc[i]*multiplier + ema[i-1]*(1-multiplier)
    return pd.Series(ema).to_frame(name="{} EMA".format(period))



def macd(price_array: np.ndarray, small_ema_period: int=12, large_ema_period: int=26, signal_line: int=9, smoothing: int=2) -> pd.DataFrame:
    """
    Takes in an array of prices and returns a dataframe with MACD, Signal Line and MACD Histogram.
    Moving Average Convergence/Divergence (MACD) describes changes in the strength, direction, momentum, and duration of a trend.
    """
    verify_array(array=price_array)
    small_ema_period = int(small_ema_period)
    large_ema_period = int(large_ema_period)
    signal_line = int(signal_line)
    smoothing = int(smoothing)
    if large_ema_period<=small_ema_period:
        raise ValueError("The argument large_ema_period must be bigger than the argument small_ema_period.")
    macd_df = pd.DataFrame(columns=["{} EMA".format(small_ema_period), "{} EMA".format(large_ema_period), "MACD", "MACD Signal Line",
                                    "MACD Histogram"])
    signal_line_mult = smoothing/(1+signal_line)
    # Small EMA
    macd_df["{} EMA".format(small_ema_period)] = ema(price_array=price_array, period=small_ema_period, smoothing=smoothing)
    # Large EMA
    macd_df["{} EMA".format(large_ema_period)] = ema(price_array=price_array, period=large_ema_period, smoothing=smoothing)
    # MACD
    macd_df["MACD"] = macd_df["{} EMA".format(small_ema_period)] - macd_df["{} EMA".format(large_ema_period)]
    # Signal Line
    macd_df.iloc[large_ema_period-2+signal_line, 3] = macd_df["MACD"].iloc[large_ema_period-1:large_ema_period-1+signal_line].mean()
    for i in range(large_ema_period-1+signal_line, len(macd_df)):
        macd_df.iloc[i, 3] = macd_df["MACD"].iloc[i]*signal_line_mult + macd_df["MACD Signal Line"].iloc[i-1]*(1-signal_line_mult)
    # MACD Histogram
    macd_df["MACD Histogram"] = macd_df["MACD"] - macd_df["MACD Signal Line"]
    return macd_df



def bollinger_bands(price_array: np.ndarray, period: int=50, n_std: int=2) -> pd.DataFrame:
    """
    Takes in an array of prices and returns a dataframe with SMA, and the upper and lowr Bolinger Bands.
    Bollinger Band is an SMA with additional upper and lower bands contain price action within n_deviations away from the SMA line.
    """
    verify_array(array=price_array)
    period = int(period)
    n_std = int(n_std)
    s = pd.Series(price_array)
    boll_df = sma(price_array=price_array, period=period)
    boll_df["Upper Band"] = boll_df["{} SMA".format(period)] + (s.rolling(window=period).std()*n_std)
    boll_df["Lower Band"] = boll_df["{} SMA".format(period)] - (s.rolling(window=period).std()*n_std)
    return boll_df



def rsi(price_array: np.ndarray, period: int=14) -> pd.DataFrame:
    """
    Takes in an array of prices and returns a dataframe of RSI readings.
    Relative Strength Index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions.
    """
    verify_array(array=price_array)
    period = int(period)
    s = pd.Series(price_array)
    rsi_df = pd.DataFrame(columns=["U", "D", "U SMMA", "D SMMA", "RS", "RSI"])
    rsi_df["U"] = np.nan*np.ones(len(s))
    for i in range(1, len(s)):
        if s.iloc[i]>=s.iloc[i-1]:
            # U
            rsi_df.iloc[i, 0] = s.iloc[i] - s.iloc[i-1]
            # D
            rsi_df.iloc[i, 1] = 0
        else:
            # U
            rsi_df.iloc[i, 0] = 0
            # D
            rsi_df.iloc[i, 1] = s.iloc[i-1] - s.iloc[i]
    # U SMMA
    rsi_df.iloc[period, 2] = rsi_df["U"].iloc[1:period+1].mean()
    # D SMMA
    rsi_df.iloc[period, 3] = rsi_df["D"].iloc[1:period+1].mean()
    for i in range(period+1, len(s)):
        # U SMMA
        rsi_df.iloc[i, 2] = (rsi_df["U SMMA"].iloc[i-1]*(period-1) + rsi_df["U"].iloc[i])/period
        # D SMMA
        rsi_df.iloc[i, 3] = (rsi_df["D SMMA"].iloc[i-1]*(period-1) + rsi_df["D"].iloc[i])/period
    rsi_df["RS"] = rsi_df["U SMMA"]/rsi_df["D SMMA"]
    rsi_df["RSI"] = 100 - 100/(1+rsi_df["RS"])
    return rsi_df



def adx(high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray, period: int=14) -> pd.DataFrame:
    """
    Takes in arrays of high, low and close prices, and returns a dataframe with +DI, -DI and ADX.
    Average Directional Index (ADX) is an indicator that describes the relative strength of the trend.
    """
    compare_array_len(array_1=close_price, array_2=high_price, array_1_name="close_price", array_2_name="high_price")
    compare_array_len(array_1=close_price, array_2=low_price, array_1_name="close_price", array_2_name="low_price")
    if len(high_price)!=len(low_price):
        raise ValueError("The arguments high_price, low_price  and close_price must be of the same length.")
    period = int(period)
    high_s = pd.Series(high_price)
    low_s = pd.Series(low_price)
    close_s = pd.Series(close_price)
    adx_df = pd.DataFrame(columns=["+DM", "-DM", "TR", "ATR", "+DM SMMA", "-DM SMMA", "+DI", "-DI",
                                                   "Abs(+DI--DI)", "ADX"])
    adx_df["+DM"] = np.nan*np.ones(len(close_s))
    # +DM and -DM
    for i in range(1, len(high_s)):
        up_move = high_s.iloc[i] - high_s.iloc[i-1]
        down_move = low_s.iloc[i-1] - low_s.iloc[i]
        if (up_move > down_move) and (up_move > 0):
            adx_df.iloc[i, 0] = up_move
        else:
            adx_df.iloc[i, 0] = 0
        if (down_move > up_move) and (down_move > 0):
            adx_df.iloc[i, 1] = down_move
        else:
            adx_df.iloc[i, 1] = 0
    # TR
    for i in range(1, len(high_s)):
        adx_df.iloc[i, 2] = max(high_s.iloc[i], close_s.iloc[i-1]) - min(low_s.iloc[i], close_s.iloc[i-1])
    # ATR
    adx_df.iloc[period-1, 3] = adx_df["TR"].iloc[0:period].mean()
    for i in range(period, len(adx_df)):
        adx_df.iloc[i, 3] = (adx_df["ATR"].iloc[i-1]*(period-1) + adx_df["TR"].iloc[i])/period
    # +DM SMMA and -DM SMMA
    adx_df.iloc[period, 4] = adx_df["+DM"].iloc[1:period+1].mean()
    adx_df.iloc[period, 5] = adx_df["-DM"].iloc[1:period+1].mean()
    for i in range(period+1, len(high_s)):
        adx_df.iloc[i, 4] = (adx_df["+DM SMMA"].iloc[i-1]*(period-1) + adx_df["+DM"].iloc[i])/period
        adx_df.iloc[i, 5] = (adx_df["-DM SMMA"].iloc[i-1]*(period-1) + adx_df["-DM"].iloc[i])/period
    # +DI and -DI
    adx_df["+DI"] = adx_df["+DM SMMA"]*(100/adx_df["ATR"])
    adx_df["-DI"] = adx_df["-DM SMMA"]*(100/adx_df["ATR"])
    # |+DI - -DI|
    adx_df["Abs(+DI--DI)"] = abs(adx_df["+DI"]-adx_df["-DI"])
    # ADX
    adx_df.iloc[2*period, 9] = adx_df["Abs(+DI--DI)"].iloc[period:2*period+1].mean()
    for i in range(2*period+1, len(adx_df)):
        adx_df.iloc[i, 9] = (adx_df["ADX"].iloc[i-1]*(period-1) + adx_df["Abs(+DI--DI)"].iloc[i])/period
    adx_df["ADX"] = adx_df["ADX"]*(100/(adx_df["+DI"]+adx_df["-DI"]))
    return adx_df