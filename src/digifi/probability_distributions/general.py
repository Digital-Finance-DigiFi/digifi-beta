import abc
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import type_check



class ProbabilityDistributionType(Enum):
    DISCRETE_DISTRIBUTION = 1
    CONTINUOUS_DISTRIBUTION = 2



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
    def pdf(self) -> np.ndarray:
        """
        Probability density function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self) -> np.ndarray:
        """
        Cummulative distribution function (CDF).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mgf(self) -> np.ndarray:
        """
        Moment generating function (MGF).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def cf(self) -> np.ndarray:
        """
        Characteristic function (CF).
        Characteristic function is the Fourier transform of the PDF.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        Inverse cumulative distribution function (CDF), else knwon as quantile function.
        """
        raise NotImplementedError



def skewness(array: np.ndarray) -> float:
    """
    ## Description
    Calculates the skewness of a given data array. Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, or undefined. Fisher-Pearson coefficient of skewness is used.

    ### Input:
        - array (np.ndarray): A numpy array for which the skewness is to be calculated.

    ### Output:
        - Skewness value (float) of the given data.

    ### Formula:
        - Skewness = \\frac{E[(X-\mu)^{3}]}{(E[(X-\mu)^{2}])^{\\frac{3}{2}}}, where X is the random variable, \mu is the mean of X, and E is the expectation operator.

    ### Links:
        - For more on skewness: https://en.wikipedia.org/wiki/Skewness
    """

    type_check(value=array, type_=np.ndarray, value_name="array")
    difference = array - np.mean(array)
    return np.mean(difference**3)/(np.mean(difference**2)**(3/2))



def kurtosis(array: np.ndarray) -> float:
    """
    ## Description
    Computes the kurtosis of a given data array. Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable. Higher kurtosis implies a heavier tail. The calculation here does not subtract 3, hence this is the 'excess kurtosis'.

    ### Input:
        - array (np.ndarray): A numpy array for which the kurtosis is to be calculated.

    ### Output:
        - Kurtosis value (float) of the given data.

    ### Formula:
        - Kurtosis = \\frac{E[(X-\mu)^{4}]}{(E[(X-\mu)^{2}])^{2}}, where X is the random variable, \mu is the mean of X, and E is the expectation operator.

    ### Links:
        - For more on kurtosis: https://en.wikipedia.org/wiki/Kurtosis
    """

    type_check(value=array, type_=np.ndarray, value_name="array")
    difference = array - np.mean(array)
    return np.mean(difference**2)/(np.mean(difference**2)**2)