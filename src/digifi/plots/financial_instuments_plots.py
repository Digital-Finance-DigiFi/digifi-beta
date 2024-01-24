from typing import Union, Callable
import numpy as np
import plotly.graph_objects as go
from .utilities import compare_array_len



def plot_option_payoff(payoff: Callable, strike_price: float, start_price: float, stop_price: float, n_prices: int=100, profit: bool=False,
                       initial_option_price: float=0, return_fig_object: bool=False) -> Union[go.Figure, None]:
    asset_prices = np.linspace(start=float(start_price), stop=float(stop_price), num=int(n_prices))
    payoffs = payoff(s_t=asset_prices, k=strike_price)
    if profit:
        payoffs = payoffs - initial_option_price
    fig = go.Figure(go.Scatter(x=asset_prices, y=payoffs, name="Option Payoff"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_option_value_surface(price_array: np.ndarray, times_to_maturity: np.ndarray, option_pv_matrix: np.ndarray,
                              return_fig_object: bool=False) -> Union[go.Figure, None]:
    compare_array_len(array_1=times_to_maturity, array_2=option_pv_matrix, array_1_name="times_to_maturity", array_2_name="option_pv_matrix")
    compare_array_len(array_1=price_array, array_2=option_pv_matrix[0], array_1_name="price_array", array_2_name="option_pv_matrix[0]")
    fig = go.Figure(go.Surface(x=price_array, y=times_to_maturity, z=option_pv_matrix, showscale=False))
    fig.update_layout(title="Option Value Surface")
    fig.update_scenes(xaxis_title_text="Asset Price", yaxis_title_text="Time to Maturity", zaxis_title_text="Option Value")
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None