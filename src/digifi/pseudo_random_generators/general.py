import abc
import numpy as np



class PseudoRandomGeneratorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "generate") and
                callable(subclass.generate))
    
    @abc.abstractmethod
    def generate(self) -> np.ndarray:
        """
        Array of generated pseudo-random numbers. 
        """
        raise NotImplementedError
