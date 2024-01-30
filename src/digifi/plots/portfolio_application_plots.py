from typing import Union
import numpy as np
import plotly.graph_objects as go
from .utilities import type_check



def plot_portfolio_cumulative_returns(cumulative_portfolio_returns: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    """
    Plot the cumulative return of the portfolio.
    """
    type_check(value=cumulative_portfolio_returns, type_=np.ndarray, value_name="cumulative_portfolio_returns")
    time_steps = np.arange(start=0, stop=len(cumulative_portfolio_returns), step=1)
    fig = go.Figure(go.Scatter(x=time_steps, y=100*cumulative_portfolio_returns, name="Portfolio Performance (%)"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_efficient_frontier(efficient_frontier_dict: dict, return_fig_object: bool=False) -> Union[go.Figure, None]:
    """
    Plot efficient frontier along with maximum Sharpe ratio point and minimum volatility point.
    """
    # Minimum volatility portfolio
    min_vol = go.Scatter(x=[100*efficient_frontier_dict["min_vol"]["std"]], y=[100*efficient_frontier_dict["min_vol"]["return"]], name="Minimum Volatility Portfolio",
                         mode="markers", marker=dict(color="red", size=14, line=dict(width=3, color="black")))
    # Maximum Sharpe ratio portfolio
    max_sr = go.Scatter(x=[100*efficient_frontier_dict["max_sr"]["std"]], y=[100*efficient_frontier_dict["max_sr"]["return"]], name="Maximum Sharpe Ratio Portfolio",
                         mode="markers", marker=dict(color="green", size=14, line=dict(width=3, color="black")))
    # Efficient frontier
    ef_curve = go.Scatter(x=100*efficient_frontier_dict["eff"]["std"], y=100*efficient_frontier_dict["eff"]["return"], name="Efficient Frontier",
                          mode="lines", line=dict(color="black", width=4, dash="dashdot"))
    plots = [min_vol, max_sr, ef_curve]
    layout = go.Layout(title = "Portfolio Optimisation with the Efficient Frontier", width=800, height=600,
                       yaxis = dict(title="Return (%)"), xaxis = dict(title="Volatility (%)"))
    fig = go.Figure(data=plots, layout=layout)
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None