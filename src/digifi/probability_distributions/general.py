import abc
from enum import Enum
import numpy as np
from digifi.utilities.general_utils import type_check



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
        ## Description
        Probability density function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self) -> np.ndarray:
        """
        ## Description
        Cummulative distribution function (CDF).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mgf(self) -> np.ndarray:
        """
        ## Description
        Moment generating function (MGF).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def cf(self) -> np.ndarray:
        """
        ## Description
        Characteristic function (CF).\n
        Characteristic function is the Fourier transform of the PDF.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Inverse cumulative distribution function (CDF), else knwon as quantile function.
        """
        raise NotImplementedError



def skewness(array: np.ndarray) -> float:
    """
    ## Description
    Calculates the skewness of a given data array. Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean.\n
    The skewness value can be positive, zero, negative, or undefined.\n
    Fisher-Pearson coefficient of skewness is used.
    ### Input:
        - array (np.ndarray): An array for which the skewness is to be calculated
    ### Output:
        - Skewness value (float) of the given data
    ### LaTeX Formula:
        - \\textit{Skewness} = \\frac{E[(X-\\mu)^{3}]}{(E[(X-\\mu)^{2}])^{\\frac{3}{2}}}

    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Skewness
        - Original Source: N/A
    """
    type_check(value=array, type_=np.ndarray, value_name="array")
    difference = array - np.mean(array)
    return np.mean(difference**3)/(np.mean(difference**2)**(3/2))



def kurtosis(array: np.ndarray) -> float:
    """
    ## Description
    Computes the kurtosis of a given data array. Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.\n
    Higher kurtosis implies a heavier tail.\n
    The calculation here does not subtract 3, hence this is the 'excess kurtosis'.
    ### Input:
        - array (np.ndarray): An array for which the kurtosis is to be calculated
    ### Output:
        - Kurtosis value (float) of the given data
    ### LaTeX Formula:
        - \\textit{Kurtosis} = \\frac{E[(X-\\mu)^{4}]}{(E[(X-\\mu)^{2}])^{2}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Kurtosis
        - Original SOurce: N/A
    """
    type_check(value=array, type_=np.ndarray, value_name="array")
    difference = array - np.mean(array)
    return np.mean(difference**2)/(np.mean(difference**2)**2)