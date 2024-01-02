import numpy as np
from src.digifi.utilities.general_utils import (verify_array, compare_array_len, rolling)



def sma(price_array: np.ndarray, period: int=15) -> np.ndarray:
    """
    Takes in an array of prices and returns an array of SMA readings.
    Simple Moving Average (SMA) describes the direction of the trend, and is computed using the mean over the certain window of readings.
    """
    verify_array(array=price_array, array_name="price_array")
    return np.append(np.nan*np.ones(int(period-1)), np.mean(rolling(array=price_array, window=int(period)), axis=1))



def ema(price_array: np.ndarray, period: int=20, smoothing: int=2) -> np.ndarray:
    """
    Takes in an array of prices and returns an array of EMA readings.
    Exponential Moving Average (EMA) describes the direction of the trend, and requires previous EMA and the latest price to compute;
    the first EMA reading will be same as SMA.
    """
    verify_array(array=price_array, array_name="price_array")
    period = int(period)
    smoothing = int(smoothing)
    multiplier = smoothing/(1+period)
    ema = np.nan*np.ones(len(price_array))
    ema[period-1] = np.mean(price_array[0:period])
    for i in range(period, len(price_array)):
        ema[i] = price_array[i]*multiplier + ema[i-1]*(1-multiplier)
    return ema



def macd(price_array: np.ndarray, small_ema_period: int=12, large_ema_period: int=26, signal_line: int=9, smoothing: int=2) -> dict:
    """
    Takes in an array of prices and returns a dataframe with MACD, Signal Line and MACD Histogram.
    Moving Average Convergence/Divergence (MACD) describes changes in the strength, direction, momentum, and duration of a trend.
    """
    verify_array(array=price_array, array_name="price_array")
    small_ema_period = int(small_ema_period)
    large_ema_period = int(large_ema_period)
    signal_line = int(signal_line)
    smoothing = int(smoothing)
    if large_ema_period<=small_ema_period:
        raise ValueError("The argument large_ema_period must be bigger than the argument small_ema_period.")
    signal_line_mult = smoothing/(1+signal_line)
    # Small EMA
    small_ema = ema(price_array=price_array, period=small_ema_period, smoothing=smoothing)
    # Large EMA
    large_ema = ema(price_array=price_array, period=large_ema_period, smoothing=smoothing)
    # MACD
    macd = small_ema - large_ema
    # Signal Line
    signal_line_ = np.nan * np.ones(len(price_array))
    signal_line_[large_ema_period-2+signal_line] = np.mean(macd[large_ema_period-1:large_ema_period-1+signal_line])
    for i in range(large_ema_period-1+signal_line, len(small_ema)):
        signal_line_[i] = macd[i]*signal_line_mult + signal_line_[i-1]*(1-signal_line_mult)
    # MACD Histogram
    macd_hist = macd - signal_line_
    return {"small_ema":small_ema, "large_ema":large_ema, "macd":macd, "signal_line":signal_line_, "macd_hist":macd_hist}



def bollinger_bands(price_array: np.ndarray, period: int=50, n_std: int=2) -> dict:
    """
    Takes in an array of prices and returns a dataframe with SMA, and the upper and lowr Bolinger Bands.
    Bollinger Band is an SMA with additional upper and lower bands contain price action within n_deviations away from the SMA line.
    """
    verify_array(array=price_array, array_name="price_array")
    period = int(period)
    n_std = int(n_std)
    sma_ = sma(price_array=price_array, period=period)
    deviation = np.append(np.nan*np.ones(int(period-1)), np.std(rolling(array=price_array, window=int(period)), axis=1)*n_std)
    return {"sma":sma_, "upper_band":sma_+deviation, "lower_band":sma_-deviation}



def rsi(price_array: np.ndarray, period: int=14, oversold_band: float=30, overbought_band: float=70) -> dict:
    """
    Takes in an array of prices and returns a dataframe of RSI readings.
    Relative Strength Index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions.
    """
    verify_array(array=price_array, array_name="price_array")
    period = int(period)
    price_array_length = len(price_array)
    # rsi_df = pd.DataFrame(columns=["U", "D", "U SMMA", "D SMMA", "RS", "RSI"])
    rsi_u = np.nan * np.ones(price_array_length)
    rsi_d = np.nan * np.ones(price_array_length)
    rsi_u_smma = np.nan * np.ones(price_array_length)
    rsi_d_smma = np.nan * np.ones(price_array_length)
    for i in range(1, price_array_length):
        if price_array[i]>=price_array[i-1]:
            # U
            rsi_u[i] = price_array[i] - price_array[i-1]
            # D
            rsi_d[i] = 0
        else:
            # U
            rsi_u[i] = 0
            # D
            rsi_d[i] = price_array[i-1] - price_array[i]
    # U SMMA
    rsi_u_smma[period] = np.mean(rsi_u[1:period+1])
    # D SMMA
    rsi_d_smma[period] = np.mean(rsi_d[1:period+1])
    for i in range(period+1, price_array_length):
        # U SMMA
        rsi_u_smma[i] = (rsi_u_smma[i-1]*(period-1) + rsi_u[i])/period
        # D SMMA
        rsi_d_smma[i] = (rsi_d_smma[i-1]*(period-1) + rsi_d[i])/period
    rs = rsi_u_smma/rsi_d_smma
    rsi = 100 - 100/(1+rs)
    return {"u":rsi_u, "d":rsi_d, "u_smma":rsi_u_smma, "rsi_d_smma":rsi_d_smma, "rs":rs, "rsi":rsi,
            "oversold":oversold_band*np.ones(len(price_array)), "overbought":overbought_band*np.ones(len(price_array))}



def adx(high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray, period: int=14, benchmark: int=25) -> dict:
    """
    Takes in arrays of high, low and close prices, and returns a dataframe with +DI, -DI and ADX.
    Average Directional Index (ADX) is an indicator that describes the relative strength of the trend.
    """
    compare_array_len(array_1=close_price, array_2=high_price, array_1_name="close_price", array_2_name="high_price")
    compare_array_len(array_1=close_price, array_2=low_price, array_1_name="close_price", array_2_name="low_price")
    period = int(period)
    price_array_length = len(close_price)
    pdm = np.nan*np.ones(price_array_length)
    mdm = np.nan*np.ones(price_array_length)
    tr = np.nan*np.ones(price_array_length)
    atr = np.nan*np.ones(price_array_length)
    pdm_smma = np.nan*np.ones(price_array_length)
    mdm_smma = np.nan*np.ones(price_array_length)
    adx_ = np.nan*np.ones(price_array_length)
    # +DM and -DM
    for i in range(1, price_array_length):
        up_move = high_price[i] - high_price[i-1]
        down_move = low_price[i-1] - low_price[i]
        if (up_move > down_move) and (up_move > 0):
            pdm[i] = up_move
        else:
            pdm[i] = 0
        if (down_move > up_move) and (down_move > 0):
            mdm[i] = down_move
        else:
            mdm[i] = 0
    # TR
    for i in range(1, price_array_length):
        tr[i] = max(high_price[i], close_price[i-1]) - min(low_price[i], close_price[i-1])
    # ATR
    atr[period-1] = np.nanmean(tr[0:period])
    for i in range(period, price_array_length):
        atr[i] = (atr[i-1]*(period-1) + tr[i])/period
    # +DM SMMA and -DM SMMA
    pdm_smma[period] = np.mean(pdm[1:period+1])
    mdm_smma[period] = np.mean(mdm[1:period+1])
    for i in range(period+1, price_array_length):
        pdm_smma[i] = (pdm_smma[i-1]*(period-1) + pdm[i])/period
        mdm_smma[i] = (mdm_smma[i-1]*(period-1) + mdm[i])/period
    # +DI and -DI
    pdi = pdm_smma*(100/atr)
    mdi = mdm_smma*(100/atr)
    # |+DI - -DI|
    abs_pdi_mdi = abs(pdi - mdi)
    # ADX
    adx_[2*period] = np.mean(abs_pdi_mdi[period:2*period + 1])
    for i in range(2*period+1, price_array_length):
        adx_[i] = (adx_[i-1]*(period-1) + abs_pdi_mdi[i])/period
    adx_ =adx_*(100/(pdi + mdi))
    return {"pdm":pdm, "mdm":mdm, "pdi":pdi, "mdi":mdi, "adx":adx_, "benchmark":benchmark*np.ones(price_array_length)}



def obv(close_price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Takes in arrays of price and volume and returns a dataframe of OBV readings.
    On-Balance Volume (OBV) is an indicator that describes the relationship between price and volume in the market.
    """
    compare_array_len(array_1=close_price, array_2=volume, array_1_name="close_price", array_2_name="volume")
    obv = np.zeros(len(close_price))
    for i in range(1, len(close_price)):
        if close_price[i]>close_price[i-1]:
            obv[i] = obv[i-1]+volume[i]
        elif close_price[i]<close_price[i-1]:
            obv[i] = obv[i-1]-volume[i]
        else:
            obv[i] = obv[i-1]
    obv[0] = np.nan
    return obv