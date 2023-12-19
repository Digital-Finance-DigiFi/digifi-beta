from typing import Union
import numpy as np
from scipy.special import erf
from src.digifi.probability_distributions.general import (ProbabilityDistributionType, ProbabilityDistributionStruct,
                                                          ProbabilityDistributionInterface)



class ContinuousUniformDistribution(ProbabilityDistributionStruct, ProbabilityDistributionInterface):
    """
    Methods and properties of continuous uniform distribution.
    """
    def __init__(self, a: float, b: float) -> None:
        a = float(a)
        b = float(b)
        if a>b:
            raise ValueError("The argument a must be smaller or equal to the argument b.")
        # ContinuousUniformDistribution class arguments
        self.a = a
        self.b = b
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.CONTINUOUS_DISTRIBUTION
        self.mean = (a+b)/2
        self.median = (a+b)/2
        self.mode = (a+b)/2
        self.variance = ((b-a)**2)/12
        self.skewness = 0
        self.excess_kurtosis = -6/5
        self.entropy = np.log(b-a)
    
    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where((self.a<=x) and (x<=self.b), 1/(self.b-self.a), 0)
    
    def cdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where((self.a<=x), np.minimum((x-self.a)/(self.b-self.a), 1), 0)
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(t!=0, (np.exp(t*self.b)-np.exp(t*self.b))/(t*(self.b-self.a)), 1)
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(t!=0, (np.exp(1j*t*self.b)-np.exp(1j*t*self.b))/(1j*t*(self.b-self.a)), 1)



class NormalDistribution(ProbabilityDistributionStruct, ProbabilityDistributionInterface):
    """
    Methods and properties of normal distribution.
    """
    def __init__(self, mu: float, sigma: float) -> None:
        mu = float(mu)
        sigma = float(sigma)
        # NormalDistribution class arguments
        self.mu = mu
        self.sigma = sigma
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.CONTINUOUS_DISTRIBUTION
        self.mean = mu
        self.median = mu
        self.mode = mu
        self.variance = sigma**2
        self.skewness = 0
        self.excess_kurtosis = 0
        self.entropy = np.log(2*np.pi*np.e*(sigma**2))/2
    
    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(-((x-self.mu)/self.sigma)**2/2)/(self.sigma*np.sqrt(2*np.pi))
    
    def cdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        # TODO: Replace erf with custom error function in utilities.general_utils and remove scipy dependency from pyproject.toml
        return (1+erf((x-self.mu)/(self.sigma*np.sqrt(2))))/2
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(self.mu*t + 0.5*(self.sigma**2)*(t**2))
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(1j*self.mu*t + 0.5*(self.sigma**2)*(t**2))



class ExponentialDistribution(ProbabilityDistributionStruct, ProbabilityDistributionInterface):
    """
    Methods and properties of exponential distribution.
    """
    def __init__(self, lambda_: float):
        lambda_ = float(lambda_)
        if lambda_<=0:
            raise ValueError("The argument lambda_ must be positive.")
        # ExponentialDistribution class arguments
        self.lambda_ = lambda_
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.CONTINUOUS_DISTRIBUTION
        self.mean = 1/lambda_
        self.median = np.log(2)/lambda_
        self.mode = 0
        self.variance = 1/(lambda_**2)
        self.skewness = 2
        self.excess_kurtosis = 6
        self.entropy = 1 - np.log(lambda_)
    
    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return self.lambda_*np.exp(-self.lambda_*x)
    
    def cdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return 1 - np.exp(-self.lambda_*x)
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(t<self.lambda_, self.lambda_/(self.lambda_-t), np.nan)
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return self.lambda_/(self.lambda_-1j*t)



class LaplaceDistribution(ProbabilityDistributionStruct, ProbabilityDistributionInterface):
    """
    Methods and properties of Laplace distribution.
    """
    def __init__(self, mu: float, b: float) -> None:
        mu = float(mu)
        b = float(b)
        # LaplaceDistribution class arguments
        self.mu = mu
        self.b = b
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.CONTINUOUS_DISTRIBUTION
        self.mean = mu
        self.median = mu
        self.mode = mu
        self.variance = 2*b**2
        self.skewness = 0
        self.excess_kurtosis = 3
        self.entropy = np.log(2*b*np.e)
    
    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(np.abs(x-self.mu)/self.b)/(2*self.b)

    def cdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(x<=self.mu, 0.5*np.exp((x-self.mu)/self.b), 1-0.5*np.exp(-(x-self.mu)/self.b))

    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(np.abs(float(t))<1/self.b, np.exp(self.mu*t)/(1-(self.b**2)*(t**2)), np.nan)

    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.where(np.abs(float(t))<1/self.b, (np.exp(self.mu*1j*t))/(1-(self.b**2)*(t**2)), np.nan)