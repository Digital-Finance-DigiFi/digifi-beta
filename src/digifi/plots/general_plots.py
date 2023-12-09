from typing import Union
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.digifi.utilities.general_utils import generate_ohlc_price_df



def plot_candlestick_chart(open_price: np.ndarray, high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray,
                           timestamp: np.ndarray, volume: Union[np.ndarray, None], indicator_subplot: bool=False,
                           return_fig_object: bool=False) -> Union[go.Figure, None]:
    price_df = generate_ohlc_price_df(open_price=open_price, high_price=high_price, low_price=low_price, close_price=close_price,
                                      timestamp=timestamp, volume=volume)
    # Number of plots to be generated
    n_rows = 1
    row_heights = [1]
    if isinstance(volume, np.ndarray):
        n_rows = n_rows+1
    if bool(indicator_subplot):
        n_rows = n_rows+1
    # Plot scales
    match n_rows:
        case 1:
            row_heights = [1]
        case 2:
            row_heights = [0.8, 0.2]
        case 3:
            row_heights = [0.7, 0.15, 0.15]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, shared_yaxes=False, row_heights=row_heights, vertical_spacing=0.05)
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=price_df.index, open=price_df["Open"], high=price_df["High"], low=price_df["Low"],
                                 close=price_df["Close"], name="Candlestick Chart"), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Volume plot
    if isinstance(volume, np.ndarray):
        fig.add_trace(go.Bar(x=price_df.index, y=price_df["Volume"], name="Volume"), row=n_rows, col=1)
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None