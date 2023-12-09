from typing import Union
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.digifi.utilities.general_utils import compare_array_len
from src.digifi.technical_indicators.technical_indicators import (sma, ema, macd, bollinger_bands, rsi, adx)



def plot_sma(fig: Union[go.Figure, None], price_array: np.ndarray, timestamp: np.ndarray, period: int=15,
             return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=price_array, array_2=timestamp, array_1_name="price_array", array_2_name="timestamp")
    sma_df = sma(price_array=price_array, period=int(period))
    sma_name = "{} SMA".format(int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=sma_df[sma_name], name=sma_name), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=sma_df[sma_name], name=sma_name))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_ema(fig: Union[go.Figure, None], price_array: np.ndarray, timestamp: np.ndarray, period: int=15, smoothing: int=2,
             return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=price_array, array_2=timestamp, array_1_name="price_array", array_2_name="timestamp")
    ema_df = ema(price_array=price_array, period=int(period), smoothing=int(smoothing))
    ema_name = "{} EMA".format(int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=ema_df[ema_name], name=ema_name), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=ema_df[ema_name], name=ema_name))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_macd(fig: Union[go.Figure, None], price_array: np.ndarray, timestamp: np.ndarray, small_ema_period: int=12,
              large_ema_period: int=26, signal_line: int=9, smoothing: int=2, return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=price_array, array_2=timestamp, array_1_name="price_array", array_2_name="timestamp")
    macd_df = macd(price_array=price_array, small_ema_period=int(small_ema_period), large_ema_period=int(large_ema_period),
                   signal_line=int(signal_line), smoothing=int(smoothing))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=macd_df["MACD"], line_color="blue", name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=macd_df["MACD Signal Line"], line_color="red", name="MACD Signal Line"), row=2, col=1)
        fig.add_trace(go.Bar(x=timestamp, y=macd_df["MACD Histogram"],  name="MACD Histogram"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=macd_df["MACD"], name="MACD"))
        fig.add_trace(go.Scatter(x=timestamp, y=macd_df["MACD Signal Line"], line_color="red", name="MACD Signal Line"))
        fig.add_trace(go.Bar(x=timestamp, y=macd_df["MACD Histogram"], name="MACD Histogram"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_bollinger_bands(fig: Union[go.Figure, None], price_array: np.ndarray, timestamp: np.ndarray, period: int=50,
                         n_std: int=2, return_fig_object: bool=False) -> Union[go.Figure, None]:
    period = int(period)
    compare_array_len(array_1=price_array, array_2=timestamp, array_1_name="price_array", array_2_name="timestamp")
    boll_df = bollinger_bands(price_array=price_array, period=period, n_std=int(n_std))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=boll_df["{} SMA".format(period)], line_color="blue", name="{} SMA".format(period)), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=boll_df["Upper Band"], line_color="red", name="Upper Band"), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=boll_df["Lower Band"], line_color="green", name="Lower Band"), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=boll_df["{} SMA".format(period)], line_color="blue", name="{} SMA".format(period)))
        fig.add_trace(go.Scatter(x=timestamp, y=boll_df["Upper Band"], line_color="red", name="Upper Band"))
        fig.add_trace(go.Scatter(x=timestamp, y=boll_df["Lower Band"], line_color="green", name="Lower Band"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_rsi(fig: Union[go.Figure, None], price_array: np.ndarray, timestamp: np.ndarray, period: int=14, oversold_band: float=30,
             overbought_band: float=70, return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=price_array, array_2=timestamp, array_1_name="price_array", array_2_name="timestamp")
    rsi_df = rsi(price_array=price_array, period=int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=rsi_df["RSI"], line_color="blue", name="RSI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=oversold_band*np.ones(len(price_array)), line_color="green", name="Oversold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=overbought_band*np.ones(len(price_array)), line_color="red", name="Overbought"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=rsi_df["RSI"], line_color="blue", name="RSI"))
        fig.add_trace(go.Scatter(x=timestamp, y=oversold_band*np.ones(len(price_array)), line_color="green", name="Oversold"))
        fig.add_trace(go.Scatter(x=timestamp, y=overbought_band*np.ones(len(price_array)), line_color="red", name="Overbought"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_adx(fig: Union[go.Figure, None], high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray, timestamp: np.ndarray,
             period: float=14, benchmark: int=25, return_fig_object: bool=False) -> Union[go.Figure, None]:
    benchmark = float(benchmark)
    compare_array_len(array_1=close_price, array_2=timestamp, array_1_name="close_price", array_2_name="timestamp")
    adx_df = adx(high_price=high_price, low_price=low_price, close_price=close_price, period=int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=timestamp, y=adx_df["ADX"], line_color="blue", name="ADX"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=adx_df["-DI"], line_color="red", name="-DI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=adx_df["+DI"], line_color="green", name="+DI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamp, y=benchmark*np.ones(len(close_price)), line_color="black", name="Benchmark"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=timestamp, y=adx_df["ADX"], line_color="blue", name="ADX"))
        fig.add_trace(go.Scatter(x=timestamp, y=adx_df["-DI"], line_color="red", name="-DI"))
        fig.add_trace(go.Scatter(x=timestamp, y=adx_df["+DI"], line_color="green", name="+DI"))
        fig.add_trace(go.Scatter(x=timestamp, y=benchmark*np.ones(len(close_price)), line_color="black", name="Benchmark"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None