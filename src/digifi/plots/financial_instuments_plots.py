from typing import Union
import numpy as np
import plotly.graph_objects as go
from src.digifi.utilities.general_utils import compare_array_len



def plot_option_payoff(asset_prices: np.ndarray, payoffs: np.ndarray, return_fig_object: bool = False) -> Union[go.Figure, None]:
    compare_array_len(array_1=asset_prices, array_2=payoffs, array_1_name="asset_prices", array_2_name="payoffs")
    fig = go.Figure(go.Scatter(x=asset_prices, y=payoffs, name="Option Payoff"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None