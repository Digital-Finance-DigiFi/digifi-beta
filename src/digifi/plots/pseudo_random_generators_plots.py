from typing import Union
import numpy as np
import plotly.graph_objects as go



def plot_pdf(points: np.ndarray, n_bins: int=100, return_fig_object: bool=False) -> Union[go.Figure, None]:
    fig = go.Figure(go.Histogram(x=points, histnorm="probability density", nbinsx=int(n_bins), name="Probability Density Function"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None



def plot_2d_scattered_points(points: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    x = points[0:len(points)-1]
    y = points[1:len(points)]
    fig = go.Figure(go.Scatter(x=x, y=y, name="2D Scatter Plot", mode="markers"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None

    

def plot_3d_scattered_points(points: np.ndarray, return_fig_object: bool=False) -> Union[go.Figure, None]:
    x = points[0:len(points)-2]
    y = points[1:len(points)-1]
    z = points[2:len(points)]
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, name="3D Scatter Plot", mode="markers"))
    if bool(return_fig_object):
        return fig
    else:
        fig.show()
        return None