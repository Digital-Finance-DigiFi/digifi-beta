from typing import Union, Callable
import numpy as np
import plotly.graph_objects as go



def plot_option_payoff(payoff: Callable, strike_price: float, start_price: float, stop_price: float, n_prices: int=100, profit: bool=False,
                       initial_option_price: float=0, return_fig_object: bool = False) -> Union[go.Figure, None]:
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



# TODO: Add option 3D surface plot