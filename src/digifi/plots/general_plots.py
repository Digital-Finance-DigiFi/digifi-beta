from typing import Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utilities import compare_array_len



def plot_candlestick_chart(open_price: np.ndarray, high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray,
                           time_array: np.ndarray, volume: Union[np.ndarray, None], indicator_subplot: bool=False,
                           return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=close_price, array_2=open_price, array_1_name="close_price", array_2_name="open_price")
    compare_array_len(array_1=close_price, array_2=high_price, array_1_name="close_price", array_2_name="high_price")
    compare_array_len(array_1=close_price, array_2=low_price, array_1_name="close_price", array_2_name="low_price")
    compare_array_len(array_1=close_price, array_2=time_array, array_1_name="close_price", array_2_name="time_array")
    # Number of plots to be generated
    n_rows = 1
    row_heights = [1]
    if isinstance(volume, np.ndarray):
        n_rows = n_rows + 1
        compare_array_len(array_1=close_price, array_2=volume, array_1_name="close_price", array_2_name="volume")
    if bool(indicator_subplot):
        n_rows = n_rows + 1
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
    fig.add_trace(go.Candlestick(x=time_array, open=open_price, high=high_price, low=low_price, close=close_price,
                                 name="Candlestick Chart"), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Volume plot
    if isinstance(volume, np.ndarray):
        fig.add_trace(go.Bar(x=time_array, y=volume, name="Volume"), row=n_rows, col=1)
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None