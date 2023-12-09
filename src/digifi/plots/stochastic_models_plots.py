from typing import Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px



def plot_stochastic_paths(paths: np.ndarray, expected_path: Union[np.ndarray, None]=None, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """Plot of the random paths taken by a stochastic process."""
        plotting_df = pd.DataFrame(paths)
        if isinstance(expected_path, np.ndarray):
            if len(expected_path)!=len(plotting_df):
                raise ValueError("The length of argument expected_path does not equal to the length of the argument paths.")
            plotting_df = pd.concat((plotting_df, pd.Series(expected_path, name="Expected Path")), axis=1)
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None