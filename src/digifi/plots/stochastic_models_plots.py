from typing import Union
import numpy as np
import plotly.graph_objects as go
from .utilities import compare_array_len



def plot_stochastic_paths(paths: np.ndarray, expected_path: Union[np.ndarray, None]=None, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Plot of the random paths taken by a stochastic process.
        """
        time_steps = np.arange(start=0, stop=len(paths[0]), step=1)
        plots = []
        if isinstance(expected_path, np.ndarray):
            compare_array_len(array_1=paths[0], array_2=expected_path, array_1_name="paths[0]", array_2_name="expected_path")
            plots.append(go.Scatter(x=time_steps, y=expected_path, name="Expected Path"))
        for i in range(len(paths)):
             plots.append(go.Scatter(x=time_steps, y=paths[i]))
        # Relocate expected path to be the last one displayed
        plots.append(plots.pop(0))
        layout = go.Layout(title = "Stochastic Simulation", yaxis = dict(title="Value of S"), xaxis = dict(title="Time Step"))
        fig = go.Figure(data=plots, layout=layout)
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None