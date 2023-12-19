from typing import Union
import abc
import numpy as np
import plotly.graph_objects as go



class PseudoRandomGeneratorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "get_randomized_array") and
                callable(subclass.get_randomized_array) and
                hasattr(subclass, "plot_pdf") and
                callable(subclass.plot_pdf) and
                hasattr(subclass, "plot_2d_scattered_points") and
                callable(subclass.plot_2d_scattered_points) and
                hasattr(subclass, "plot_3d_scattered_points") and
                callable(subclass.plot_3d_scattered_points))
    
    @abc.abstractmethod
    def get_randomized_array(self) -> np.ndarray:
        """
        Array of generated pseudo-random numbers. 
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def plot_pdf(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Histogram plot of the probability density function.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def plot_2d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        2D scatter plot of the pseudo-random points generated.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def plot_3d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        3D scatter plot of the pseudo-random points generated.
        """
        raise NotImplementedError
