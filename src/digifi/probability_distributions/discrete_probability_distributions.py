from typing import Union
import numpy as np
from src.digifi.probability_distributions.general import (ProbabilityDistributionType, ProbabilityDistributionInterface)
from src.digifi.utilities.maths_utils import (factorial, n_choose_r)



class BernoulliDistribution(ProbabilityDistributionInterface):
    """
    Methods and properties of Bernoulli distribution.
    """
    def __init__(self, p: float) -> None:
        p = float(p)
        if (p<0) or (1<p):
            raise ValueError("The argument p must be in the [0, 1] range.")
        # BernoulliDistribution class arguments
        self.p = p
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.DISCRETE_DISTRIBUTION
        self.mean = p
        if p<0.5:
            self.meadian = 0
            self.mode = 0
        elif p<0.5:
            self.meadian = 1
            self.mode = 1
        else:
            self.meadian = (0, 1)
            self.mode = (0, 1)
        self.variance = p*(1-p)
        self.skewness = ((1-p)-p)/(np.sqrt((1-p)*p))
        self.excess_kurtosis = (1-6*(1-p)*p)/((1-p)*p)
        self.entropy = -(1-p)*np.log(1-p) - p*np.log(p)
    
    def __pdf(self, k: int) -> float:
        if int(k)==0:
            return 1-self.p
        elif int(k)==1:
            return self.p
        else:
            raise ValueError("The argument k must be in the \{0, 1\} set.")
        
    def pdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        if isinstance(k, np.ndarray):
            result = np.array([])
            for k_ in k:
                result = np.append(result, self.__pdf(k=int(k_)))
        elif isinstance(k, float):
            result = self.__pdf(k=k)
        else:
            raise TypeError("The argument k must be of either np.ndarray or int type.")
        return result
    
    def __cdf(self, k: int) -> float:
        if int(k)<0:
            return 0.0
        elif 1<=int(k):
            return 1.0
        else:
            return 1-self.p
    
    def cdf(self, k: Union[np.ndarray, int]):
        if isinstance(k, np.ndarray):
            result = np.array([])
            for k_ in k:
                result = np.append(result, self.__cdf(k=int(k_)))
        elif isinstance(k, float):
            result = self.__cdf(k=k)
        else:
            raise TypeError("The argument k must be of either np.ndarray or int type.")
        return result
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (1-self.p) + self.p*np.exp(t)
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (1-self.p) + self.p*np.exp(1j*t)



class BinomialDistribution(ProbabilityDistributionInterface):
    """
    Methods and properties of binomial distribution.
    """
    def __init__(self, n: int, p: float) -> None:
        n = int(n)
        p = float(p)
        if n<0:
            raise ValueError("The agument n must be a positive integer.")
        if (p<0) or (1<p):
            raise ValueError("The aregument p must be in the [0, 1] range.")
        # BinomialDistribution class arguments
        self.n = n
        self.p = p
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.CONTINUOUS_DISTRIBUTION
        self.mean = n*p
        self.median = (np.floor(n*p), np.ceil(n*p))
        self.mode = (np.floor((n+1)*p), np.ceil((n+1)*p)-1)
        self.variance = n*p*(1-p)
        self.skewness = ((1-p)-p)/(np.sqrt(n*p*(1-p)))
        self.excess_kurtosis = (1-6*(1-p)*p)/(n*(1-p)*p)
        self.entropy = 0.5*(np.log(2*np.pi*np.e*n*p*(1-p)))
    
    def __pdf(self, k: int) -> float:
        k = int(k)
        return n_choose_r(n=self.n, r=k)*(self.p**k)*((1-self.p)**(self.n-k))
    
    def pdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        if isinstance(k, np.ndarray):
            result = np.array([])
            for k_ in k:
                result = np.append(result, self.__pdf(k=int(k_)))
        elif isinstance(k, int):
            result = self.__pdf(k=k)
        else:
            raise TypeError("The argument k must be of either np.ndarray or int type.")
        return result
    
    def cdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        # TODO: Implement incomplete beta function from maths_utils.py
        return 0.0
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return ((1-self.p) + self.p*np.exp(t))**self.n
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return ((1-self.p) + self.p*np.exp(1j*t))**self.n



class DiscreteUniformDistribution(ProbabilityDistributionInterface):
    """
    Methods and properties of discrete uniform distribution.
    """
    def __init__(self, a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        if a>b:
            raise ValueError("The argument a must be smaller or equal to the argument b.")
        # DiscreteUniformDistribution class arguments
        self.a = a
        self.b = b
        self.n = b-a+1
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.DISCRETE_DISTRIBUTION
        self.mean = (a+b)/2
        self.median = (a+b)/2
        self.mode = None
        self.variance = (self.n**2 - 1)/12
        self.skewness = 0
        self.excess_kurtosis = -6*(self.n**2 + 1)/(5*(self.n**2 - 1))
        self.entropy = np.log(self.n)
    
    def pdf(self, n_readings: int) -> Union[np.ndarray, float]:
        return np.ones(int(n_readings))/self.n
    
    def cdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        return np.where((self.a<=k) and (k<=self.b), (np.floor(k)-self.a+1)/self.n, np.nan)
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (np.exp(self.a*t) - np.exp((self.b+1)*t))/(self.n*(1-np.exp(t)))
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (np.exp(1j*self.a*t) - np.exp(1j*(self.b+1)*t))/(self.n*(1-np.exp(1j*t)))



class PoissonDistribution(ProbabilityDistributionInterface):
    """
    Methods and properties of Poisson distribution.
    """
    def __init__(self, lambda_: float) -> None:
        lambda_ = float(lambda_)
        if lambda_<=0:
            raise ValueError("The argument lambda_ must be positive.")
        # PoissonDistribution class arguments
        self.lambda_ = lambda_
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.DISCRETE_DISTRIBUTION
        self.mean = lambda_
        self.median = np.floor(lambda_ + 1/3 + -1/(50*lambda_))
        self.mode = (np.ceil(lambda_)-1, np.floor(lambda_))
        self.variance = lambda_
        self.skewness = 1/np.sqrt(lambda_)
        self.excess_kurtosis = 1/lambda_
        self.entropy = 0.5*np.log(2*np.pi*np.e*self.lambda_) - 1/(12*self.lambda_) - 1/(24*self.lambda_**2) - 19/(360*self.lambda_**3)
    
    def __pdf(self, k: int) -> float:
        k = int(k)
        if k<0:
            raise ValueError("The argument k must be positive.")
        return ((self.lambda_**k)*np.exp(-self.lambda_))/factorial(k=k)

    def pdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        if isinstance(k, np.ndarray):
            result = np.array([])
            for k_ in k:
                result = np.append(result, self.__pdf(k=int(k_)))
        elif isinstance(k, int):
            result = self.__pdf(k=k)
        else:
            raise TypeError("The argument k must be of either np.ndarray or int type.")
        return result
    
    def __cdf(self, k: int) -> float:
        k = int(k)
        result = 0
        if k<0:
            raise ValueError("The argument k must be positive.")
        for i in range(0, np.floor(k)+1):
            result = result + (self.lambda_**i)/factorial(k=i)
        return np.exp(-self.lambda_)*result
    
    def cdf(self, k: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
        if isinstance(k, np.ndarray):
            result = np.array([])
            for k_ in k:
                result = np.append(result, self.__cdf(k=int(k_)))
        elif isinstance(k, float):
            result = self.__cdf(k=k)
        else:
            raise TypeError("The argument k must be of either np.ndarray or int type.")
        return result
    
    def mgf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(self.lambda_*(np.exp(t)-1))
    
    def cf(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.exp(self.lambda_*(np.exp(1j*t)-1))