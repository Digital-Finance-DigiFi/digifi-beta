from typing import Union
import abc
import numpy as np
import plotly.graph_objects as go
from src.digifi.plots.stochastic_models_plots import (plot_stochastic_paths)



class StochasticProcessInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "get_paths") and
                callable(subclass.get_paths) and
                hasattr(subclass, "get_expectation") and
                callable(subclass.get_expectation) and
                hasattr(subclass, "plot") and
                callable(subclass.plot))
    
    @abc.abstractmethod
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the stochastic process.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the stochastic process.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Plot of the random paths taken by the stochastic process.
        """
        raise NotImplementedError



class StochasticProcess(StochasticProcessInterface):
    # TODO: Add customizable stochastic model parts from stochastic_components module
    def __init__(self):
        pass

    def get_paths(self) -> np.ndarray:
        return super().get_paths()
    
    def get_expectation(self) -> np.ndarray:
        return super().get_expectation()
    
    def plot(self, plot_expected: bool = False, return_fig_object: bool = False) -> Union[go.Figure, None]:
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths(), expected_path=expected_path, return_fig_object=return_fig_object)