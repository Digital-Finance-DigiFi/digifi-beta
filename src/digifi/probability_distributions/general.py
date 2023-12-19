from typing import Union
import abc
from enum import Enum
from dataclasses import dataclass
import numpy as np



class ProbabilityDistributionType(Enum):
    DISCRETE_DISTRIBUTION = 1
    CONTINUOUS_DISTRIBUTION = 2



@dataclass
class ProbabilityDistributionStruct:
    """
    Struct with general probability distribution properties.
    """
    distribution_type: ProbabilityDistributionType
    mean: float
    median: float
    mode: float
    variance: float
    skewness: float
    excess_kurtosis: float
    entropy: float
    


class ProbabilityDistributionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "pdf") and
                callable(subclass.pdf) and
                hasattr(subclass, "cdf") and
                callable(subclass.cdf) and
                hasattr(subclass, "mgf") and
                callable(subclass.mgf) and
                hasattr(subclass, "cf") and
                callable(subclass.cf))
    
    @abc.abstractmethod
    def pdf(self) -> Union[np.ndarray, float]:
        """
        Probability density function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self) -> Union[np.ndarray, float]:
        """
        Cummulative distribution function (CDF).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mgf(self) -> Union[np.ndarray, float]:
        """
        Moment generating function (MGF).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def cf(self) -> Union[np.ndarray, float]:
        """
        Characteristic function (CF).
        Characteristic function is the Fourier transform of the PDF.
        """
        raise NotImplementedError
    
    # TODO: Add generate_random_points method
    # TODO: Add inverse_cdf method (i.e., F^{-1}(x)) for inverse_transformation_method function