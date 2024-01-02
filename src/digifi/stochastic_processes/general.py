import abc
import numpy as np



class StochasticProcessInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "get_paths") and
                callable(subclass.get_paths) and
                hasattr(subclass, "get_expectation") and
                callable(subclass.get_expectation))
    
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



class StochasticProcess(StochasticProcessInterface):
    # TODO: Add customizable stochastic model parts from stochastic_components module
    def __init__(self):
        pass

    def get_paths(self) -> np.ndarray:
        return super().get_paths()
    
    def get_expectation(self) -> np.ndarray:
        return super().get_expectation()