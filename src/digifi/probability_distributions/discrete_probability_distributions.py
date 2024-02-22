import numpy as np
from scipy.optimize import fsolve
from scipy.special import betainc
from digifi.utilities.general_utils import type_check
from digifi.utilities.maths_utils import (factorial, n_choose_r)
from digifi.probability_distributions.general import (ProbabilityDistributionType, ProbabilityDistributionInterface)
from digifi.probability_distributions.continuous_probability_distributions import NormalDistribution



class BernoulliDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of Bernoulli distribution.
    ### Input:
        - p (float): Probability of successful outcome
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Bernoulli_distribution
        - Original Source: N/A
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
        elif 0.5<p:
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
        
    def pdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Mass Function (PMF) for a Bernoulli distribution.
        ### Input:
            - k (np.ndarray): Array of values (0 or 1) at which to calculate the PMF
        ### Output:
            - PMF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        result = []
        for k_ in k:
            result.append(self.__pdf(k=int(k_)))
        return np.array(result)
    
    def __cdf(self, k: int) -> float:
        if int(k)<0:
            return 0.0
        elif 1<=int(k):
            return 1.0
        else:
            return 1-self.p
    
    def cdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a Bernoulli distribution.
        ### Input:
            - k (np.ndarray): Array of values (0 or 1) at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        result = []
        for k_ in k:
            result.append(self.__cdf(k=int(k_)))
        return np.array(result)
    
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Bernoulli distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return np.where(p==1.0, 1, 0)
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Moment Generating Function (MGF) for a Bernoulli distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return (1-self.p) + self.p*np.exp(t)
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Characteristic Function (CF) for a Bernoulli distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return (1-self.p) + self.p*np.exp(1j*t)



class BinomialDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of binomial distribution.
    ### Input:
        - n (int): Number of trials
        - p (float): Probability of successful outcome
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Binomial_distribution
        - Original SOurce: N/A
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
    
    def pdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Mass Function (PMF) for a binomial distribution.
        ### Input:
            - k (np.ndarray): Array of non-negative integer values at which to calculate the PMF
        ### Output:
            - PMF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        result = []
        for k_ in k:
            result.append(self.__pdf(k=int(k_)))
        return np.array(result)
    
    def cdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a binomial distribution.
        ### Input:
            - k (np.ndarray): Array of non-negative integer values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        a = self.n - np.floor(k)
        b = 1 + np.floor(k)
        return betainc(a, b, np.ones(len(k))*(1-self.p))
    
    def inverse_cdf(self, p: np.ndarray, x_0: float=0.5) -> np.ndarray:
        """
        ## Description
        Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a binomial distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        x_0 = float(x_0)
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        result = np.array([])
        for p_ in p:
            if np.isnan(p_):
                result = np.append(result, p_)
                continue
            def quantile(rv: np.ndarray) -> float:
                cdf = self.cdf(k=rv)
                return np.where(np.isnan(cdf), 0, cdf) - p_
            result = np.append(result, int(fsolve(quantile, x0=np.array([x_0]))[0]))
        return result
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Moment Generating Function (MGF) for a binomial distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return ((1-self.p) + self.p*np.exp(t))**self.n
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Characteristic Function (CF) for a binomial distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return ((1-self.p) + self.p*np.exp(1j*t))**self.n



class DiscreteUniformDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of discrete uniform distribution.
    ### Input:
        - a (int): Lower bound of the distribution
        - b (int): Upper bound of the distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Discrete_uniform_distribution
        - Original Source: N/A
    """
    def __init__(self, a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        if a>b:
            raise ValueError("The argument a must be smaller or equal to the argument b.")
        # DiscreteUniformDistribution class arguments
        self.a = a
        self.b = b
        self.n = b - a + 1
        # ProbabilityDistributionStruct arguments
        self.distribution_type = ProbabilityDistributionType.DISCRETE_DISTRIBUTION
        self.mean = (a+b)/2
        self.median = (a+b)/2
        self.mode = None
        self.variance = (self.n**2 - 1)/12
        self.skewness = 0
        self.excess_kurtosis = -6*(self.n**2 + 1)/(5*(self.n**2 - 1))
        self.entropy = np.log(self.n)
    
    def pdf(self, n_readings: int) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Mass Function (PMF) for a discrete uniform distribution.
        ### Input:
            - n_readings (int): Number of possible outcomes
        ### Output:
            - PMF values (np.ndarray) for the discrete uniform distribution
        """
        return np.ones(int(n_readings))/self.n
    
    def cdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a discrete uniform distribution.
        ### Input:
            - k (np.ndarray): Array of values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        return np.where((self.a<=k) & (k<=self.b), (np.floor(k)-self.a+1)/self.n, np.nan)
    
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a discrete uniform distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF

        ### Output:
            - Inverse CDF values (np.ndarray) for the given probabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        return (p*self.n) - 1 + self.a
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Moment Generating Function (MGF) for a discrete uniform distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return (np.exp(self.a*t) - np.exp((self.b+1)*t))/(self.n*(1-np.exp(t)))
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Characteristic Function (CF) for a discrete uniform distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return (np.exp(1j*self.a*t) - np.exp(1j*(self.b+1)*t))/(self.n*(1-np.exp(1j*t)))



class PoissonDistribution(ProbabilityDistributionInterface):
    """
    ## Description
    Methods and properties of Poisson distribution.
    ### Input:
        - lambda_ (float): Expected number of events in a given interval
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Poisson_distribution
        - Original SOurce: N/A
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
        return ((self.lambda_**k)*np.exp(-self.lambda_))/factorial(n=k)

    def pdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Probability Mass Function (PMF) for a Poisson distribution.
        ### Input:
            - k (np.ndarray): Array of non-negative integer values at which to calculate the PMF
        ### Output:
            - PMF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        result = []
        for k_ in k:
            result.append(self.__pdf(k=int(k_)))
        return np.array(result)
    
    def __cdf(self, k: int) -> float:
        k = int(k)
        result = 0
        if k<0:
            raise ValueError("The argument k must be positive.")
        for i in range(0, int(np.floor(k))):
            result = result + (self.lambda_**i)/factorial(n=i)
        return np.exp(-self.lambda_)*result
    
    def cdf(self, k: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Cumulative Distribution Function (CDF) for a Poisson distribution.
        ### Input:
            - k (np.ndarray): Array of non-negative integer values at which to calculate the CDF
        ### Output:
            - CDF values (np.ndarray) at the given k
        """
        type_check(value=k, type_=np.ndarray, value_name="k")
        result = []
        for k_ in k:
            result.append(self.__cdf(k=int(k_)))
        return np.array(result)
    
    def inverse_cdf(self, p: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Poisson distribution.
        ### Input:
            - p (np.ndarray): Probability values for which to calculate the inverse CDF
        ### Output:
            - Inverse CDF values (np.ndarray) for the givenprobabilities
        """
        type_check(value=p, type_=np.ndarray, value_name="p")
        p = np.where((p<0) | (1<p), np.nan, p)
        w = NormalDistribution(mu=0, sigma=1).inverse_cdf(p=p)
        return (self.lambda_ + np.sqrt(self.lambda_)*w + (1/3+(w**2)/6) + (-w/36 - (w**3)/72)/np.sqrt(self.lambda_)
                + (-8/405 + 7*(w**2)/810 + (w**4)/270)/self.lambda_)
    
    def mgf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Calculates the Moment Generating Function (MGF) for a Poisson distribution.
        ### Input:
            - t (np.ndarray): Input values for the MGF
        ### Output:
            - MGF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.exp(self.lambda_*(np.exp(t)-1))
    
    def cf(self, t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Computes the Characteristic Function (CF) for a Poisson distribution.
        ### Input:
            - t (np.ndarray): Input values for the CF
        ### Output:
            - CF values (np.ndarray) at the given t
        """
        type_check(value=t, type_=np.ndarray, value_name="t")
        return np.exp(self.lambda_*(np.exp(1j*t)-1))