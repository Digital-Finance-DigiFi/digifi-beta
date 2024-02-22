from typing import Union
import abc
from enum import Enum
import numpy as np



class LatticeModelPayoffType(Enum):
    LONG_CALL = 1
    LONG_PUT = 2
    CUSTOM = 3



class LatticeModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "european_option") and
                callable(subclass.european_option) and
                hasattr(subclass, "american_option") and
                callable(subclass.american_option) and
                hasattr(subclass, "bermudan_option") and
                callable(subclass.bermudan_option))

    @abc.abstractmethod
    def european_option(self) -> float:
        """
        ## Description
        Fair value of European option.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def american_option(self) -> float:
        """
        ## Description
        Fair value of American option.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def bermudan_option(self) -> float:
        """
        ## Description
        Fair value of Bermudan option.
        """
        raise NotImplementedError