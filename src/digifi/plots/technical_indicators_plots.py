from typing import Union
import numpy as np
import plotly.graph_objects as go
from .utilities import verify_array



def plot_sma(fig: Union[go.Figure, None], sma: np.ndarray, time_array: np.ndarray, period: int=15,
             return_fig_object: bool=False) -> Union[go.Figure, None]:
    verify_array(array=sma, array_name="sma")
    verify_array(array=time_array, array_name="time_array")
    sma_name = "{} SMA".format(int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=sma, name=sma_name), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=sma, name=sma_name))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_ema(fig: Union[go.Figure, None], ema: np.ndarray, time_array: np.ndarray, period: int=15, return_fig_object: bool=False) -> Union[go.Figure, None]:
    verify_array(array=ema, array_name="ema")
    verify_array(array=time_array, array_name="time_array")
    ema_name = "{} EMA".format(int(period))
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=ema, name=ema_name), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=ema, name=ema_name))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_macd(fig: Union[go.Figure, None], macd_dict: dict, time_array: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    verify_array(array=time_array, array_name="time_array")
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=macd_dict["macd"], line_color="blue", name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=macd_dict["signal_line"], line_color="red", name="MACD Signal Line"), row=2, col=1)
        fig.add_trace(go.Bar(x=time_array, y=macd_dict["macd_hist"],  name="MACD Histogram"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=macd_dict["macd"], name="MACD"))
        fig.add_trace(go.Scatter(x=time_array, y=macd_dict["signal_line"], line_color="red", name="MACD Signal Line"))
        fig.add_trace(go.Bar(x=time_array, y=macd_dict["macd_hist"], name="MACD Histogram"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_bollinger_bands(fig: Union[go.Figure, None], boll_dict: dict, time_array: np.ndarray, period: int=50,
                         return_fig_object: bool=False) -> Union[go.Figure, None]:
    period = int(period)
    verify_array(array=time_array, array_name="time_array")
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=boll_dict["sma"], line_color="blue", name="{} SMA".format(period)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=boll_dict["upper_band"], line_color="red", name="Upper Band"), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=boll_dict["lower_band"], line_color="green", name="Lower Band"), row=1, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=boll_dict["sma"], line_color="blue", name="{} SMA".format(period)))
        fig.add_trace(go.Scatter(x=time_array, y=boll_dict["upper_band"], line_color="red", name="Upper Band"))
        fig.add_trace(go.Scatter(x=time_array, y=boll_dict["lower_band"], line_color="green", name="Lower Band"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_rsi(fig: Union[go.Figure, None], rsi_dict: dict, time_array: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    verify_array(array=time_array, array_name="time_array")
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=rsi_dict["rsi"], line_color="blue", name="RSI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=rsi_dict["oversold"], line_color="green", name="Oversold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=rsi_dict["overbought"], line_color="red", name="Overbought"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=rsi_dict["rsi"], line_color="blue", name="RSI"))
        fig.add_trace(go.Scatter(x=time_array, y=rsi_dict["oversold"], line_color="green", name="Oversold"))
        fig.add_trace(go.Scatter(x=time_array, y=rsi_dict["overbought"], line_color="red", name="Overbought"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_adx(fig: Union[go.Figure, None], adx_dict: dict, time_array: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    verify_array(array=time_array, array_name="time_array")
    if isinstance(fig, go.Figure):
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["adx"], line_color="blue", name="ADX"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["mdi"], line_color="red", name="-DI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["pdi"], line_color="green", name="+DI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["benchmark"], line_color="black", name="Benchmark"), row=2, col=1)
    else:
        fig = go.Figure(go.Scatter(x=time_array, y=adx_dict["adx"], line_color="blue", name="ADX"))
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["mdi"], line_color="red", name="-DI"))
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["pdi"], line_color="green", name="+DI"))
        fig.add_trace(go.Scatter(x=time_array, y=adx_dict["benchmark"], line_color="black", name="Benchmark"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None