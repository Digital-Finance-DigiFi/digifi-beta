import numpy as np
from scipy.special import erfinv
from digifi.utilities.general_utils import type_check
from digifi.utilities.maths_utils import erf
from digifi.probability_distributions.general import (ProbabilityDistributionType, ProbabilityDistributionInterface)



class ContinuousUniformDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of continuous uniform distribution.
    ### Input:
        - a (float): Lower bound of the distribution
        - b (float): Upper bound of the distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        - Original Source: N/A
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
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Density Function (PDF) for a continuous uniform distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the PDF
        ### Output:
            - PDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return np.where((self.a<=x) and (x<=self.b), 1/(self.b-self.a), 0)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a continuous uniform distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Cumulative_distribution_function
            - Original Sourcew: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return np.where((self.a<=x), np.minimum((x-self.a)/(self.b-self.a), 1), 0)
    
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a continuous uniform distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return self.a + p*(self.b-self.a)

    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Moment Generating Function (MGF) for a continuous uniform distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.where(t!=0, (np.exp(t*self.b)-np.exp(t*self.b))/(t*(self.b-self.a)), 1)

    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Characteristic Function (CF) for a continuous uniform distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.where(t!=0, (np.exp(1j*t*self.b)-np.exp(1j*t*self.b))/(1j*t*(self.b-self.a)), 1)



class NormalDistribution(ProbabilityDistributionInterface):
    """
    Methods and properties of normal distribution.
    ### Input:
        - mu (float): Mean of the distribution
        - sigma (float): Standard deviation of the distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution
        - Original Source: N/A
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
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Density Function (PDF) of a normal distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the PDF
        ### Output:
            - PDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return np.exp(-((x-self.mu)/self.sigma)**2/2)/(self.sigma*np.sqrt(2*np.pi))
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a normal distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
            - Original SOurce: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return (1+erf((x-self.mu)/(self.sigma*np.sqrt(2))))/2

    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a normal distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return self.mu + self.sigma*np.sqrt(2)*erfinv(2*p - 1)
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Moment Generating Function (MGF) for a normal distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.exp(self.mu*t + 0.5*(self.sigma**2)*(t**2))
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Characteristic Function (CF) for a normal distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.exp(1j*self.mu*t + 0.5*(self.sigma**2)*(t**2))



class ExponentialDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of exponential distribution.
    ### Input:
        - lambda_ (float): Rate parameter
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution
        - Original Source: N/A
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
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Density Function (PDF) for an exponential distribution
        ### Input:
            - x (np.ndarray): Values at which to calculate the PDF
        ### Output:
            - PDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution#Probability_density_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return self.lambda_*np.exp(-self.lambda_*x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for an exponential distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution#Cumulative_distribution_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return 1 - np.exp(-self.lambda_*x)
    
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for an exponential distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return -np.log(1-p) / self.lambda_
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Moment Generating Function (MGF) for an exponential distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.where(t<self.lambda_, self.lambda_/(self.lambda_-t), np.nan)
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Characteristic Function (CF) for an exponential distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return self.lambda_/(self.lambda_-1j*t)



class LaplaceDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of Laplace distribution.

    ### Input:
        - mu (float): Location parameter, which is the peak of the distribution
        - b (float): Scale parameter, which controls the spread of the distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution
        - Original Source: N/A
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

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Density Function (PDF) for a Laplace distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the PDF
        ### Output:
            - PDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution#Probability_density_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return np.exp(np.abs(x-self.mu)/self.b)/(2*self.b)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a Laplace distribution.
        ### Input:
            - x (np.ndarray): Values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given x
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution#Cumulative_distribution_function
            - Original Source: N/A
        """
        type_check(value=x, type_=np.ndarray, value_name="x")
        return np.where(x<=self.mu, 0.5*np.exp((x-self.mu)/self.b), 1-0.5*np.exp(-(x-self.mu)/self.b))

    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Laplace distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return self.mu - self.b*np.sign(p-0.5)*np.log(1-2*np.abs(p-0.5))

    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Moment Generating Function (MGF) for a Laplace distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.where(np.abs(t)<1/self.b, np.exp(self.mu*t)/(1-(self.b**2)*(t**2)), np.nan)

    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Characteristic Function (CF) for a Laplace distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.where(np.abs(t)<1/self.b, (np.exp(self.mu*1j*t))/(1-(self.b**2)*(t**2)), np.nan)