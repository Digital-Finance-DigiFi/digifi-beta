from typing import Union
import abc
from enum import Enum
import numpy as np



class LatticeModelPayoffType(Enum):
    CALL = 1
    PUT = 2



class LatticeModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "call_payoff") and
                callable(subclass.call_payoff) and
                hasattr(subclass, "put_payoff") and
                callable(subclass.put_payoff) and
                hasattr(subclass, "european_option") and
                callable(subclass.european_option) and
                hasattr(subclass, "american_option") and
                callable(subclass.american_option) and
                hasattr(subclass, "bermudan_option") and
                callable(subclass.bermudan_option))
    
    @abc.abstractmethod
    def call_payoff(self, s_t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Call payoff.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def put_payoff(self, s_t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Put payoff.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def european_option(self) -> float:
        """
        Fair value of European option.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def american_option(self) -> float:
        """
        Fair value of American option.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def bermudan_option(self) -> float:
        """
        Fair value of Bermudan option.
        """
        raise NotImplementedError