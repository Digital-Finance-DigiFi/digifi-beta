from typing import Any
from enum import Enum
import numpy as np
from digifi.utilities.general_utils import type_check
from digifi.stochastic_processes.general import StochasticProcessInterface



class FellerSquareRootProcessMethod(Enum):
    """
    ## Description
    Enumeration class for different methods of simulating the Feller Square-Root Process.
    """
    EULER_MARUYAMA = 1
    ANALYTIC_EULER_MARUYAMA = 2
    EXACT = 3


class ArithmeticBrownianMotion(StochasticProcessInterface):
    """
    ## Description
    Arithmetic Brownian motion.
    ### Input:
        - mu (float): Mean of the process
        - sigma (float): Standard deviation of the process
        - n_paths (int): Number of paths to generate
        - n_steps (int): Number of steps
        - T (float): Final time step
        - s_0 (float): Initial value of the stochastic process
    ### LaTeX Formula:
        - dS_{t} = \\mu*dt + \\sigma*dW_{t}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#:~:text=solution%20claimed%20above.-,Arithmetic%20Brownian%20Motion,-%5Bedit%5D
        - Original Source: https://doi.org/10.24033/asens.476
    """
    def __init__(self, mu: float=0.05, sigma: float=0.4, n_paths: int=100, n_steps: int=200, T: float=1.0, s_0: float=100.0) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
        
    def get_paths(self) -> np.ndarray[Any, np.ndarray]:
        """
        ## Description
        Generates simulation paths for the Arithmetic Brownian Motion using the Euler-Maruyama method.
        ### Output:
            - An array (np.ndarray) of simulated paths following the Arithmetic Brownian Motion
        ### LaTeX Formula:
            - dS_{t} = \\mu dt + \\sigma dW_{t}
        """
        # Stochastic process
        dW = np.sqrt(self.dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = self.mu*self.dt + self.sigma*dW
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=1)
        return np.cumsum(dS, axis=1)
    
    def get_expectation(self) -> np.ndarray:
        """
        ## Description
        Calculates the expected path of the Arithmetic Brownian Motion
        ### Output:
            - An array (np.ndarray) of expected values of the stock price at each time step
        ### LaTeX Formula:
            - E[S_t] = \\mu t + S_{0}
        """
        return self.mu*self.t + self.s_0
    
    def get_variance(self) -> np.ndarray:
        """
        ## Description
        Calculates the variance of the Arithmetic Brownian Motion at each time step.
        ### Output:
            - An array (np.ndarray) of variances of the stock price at each time step
        ### LaTeX Formula:
            - Var[S_{t}] = \\sigma^{2} t
        """
        return self.t*(self.sigma**2)
    
    def get_auto_cov(self, index_t1: int, index_t2: int) -> np.ndarray:
        """
        ## Description
        Calculates the auto-covariance of the Arithmetic Brownian Motion between two time points.
        ### Input:
            - index_t1 (int): Index of the first time point
            - index_t2 (int): Index of the second time point
        ### Output:
            - Auto-covariance of the process between times t1 and t2 (np.ndarray)
        ### LaTeX Formula:
            - \\textit{Cov}(S_{t_{1}}, S_{t_{2}}) = \\sigma^{2} \\min(t_{1}, t_{2})
        """
        return (self.sigma**2)*min(self.t[int(index_t1)], self.t[int(index_t2)])


class GeometricBrownianMotion(StochasticProcessInterface):
    """
    ## Description
    Model describing the evolution of stock prices.
    ### Input:
        - mu (float): Mean of the process
        - sigma (float): Standard deviation of the process
        - n_paths (int): Number of paths to generate
        - n_steps (int): Number of steps
        - T (float): Final time step
        - s_0 (float): Initial value of the stochastic process
    ### LaTeX Formula:
        - dS_{t} = \\mu*S_{t}*dt + \\sigma*S_{t}*dW_{t}\n
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
        - Original Source: http://dx.doi.org/10.1086/260062
    """
    def __init__(self, mu: float=0.2, sigma: float=0.4, n_paths: int=100, n_steps: int=200, T: float=1.0, s_0: float=100.0) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
    
    def get_paths(self) -> np.ndarray[Any, np.ndarray]:
        """
        ## Description
        Simulates paths of the Geometric Brownian Motion using the Euler-Maruyama method. This method provides an approximation of the continuous-time process.
        ### Output:
            - An array (np.ndarray) of simulated stock prices following the Geometric Brownian Motion for each path and time step
        ### LaTeX Formula:
            - dS_{t} = \\mu S_{t} dt + \\sigma S_{t} dW_{t}
        """
        # Stochastic process
        dW = np.sqrt(self.dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = (self.mu-0.5*self.sigma**2)*self.dt + self.sigma*dW
        # Data formatting
        dS = np.insert(dS, 0, 0, axis=1)
        s = np.cumsum(dS, axis=1)
        return (self.s_0*np.exp(s))
    
    def get_expectation(self) -> np.ndarray:
        """
        ## Description
        Calculates the expected path of the Geometric Brownian Motion. This represents the mean trajectory of the stock price over time.
        ### Output:
            - An array (np.ndarray) of expected values of the stock price at each time step, representing the mean trajectory
        ### LaTeX Formula:
            - E[S_t] = S_{0} e^{\\mu t}
        """
        return self.s_0*np.exp(self.mu*self.t)
    
    def get_variance(self) -> np.ndarray:
        """
        ## Description
        Computes the variance of the stock price at each time step under the Geometric Brownian Motion model. This provides an indication of the variability or risk associated with the stock price.
        ### Output:
            - An array (np.ndarray) of variances of the stock price at each time step
        ### LaTeX Formula:
            - \\textit{Var}[S_{t}] = (S^{2}_{0}) e^{2\\mu t} (e^{\\sigma^{2} t} - 1)
        """
        return (self.s_0**2)*np.exp(2*self.mu*self.t)*(np.exp(self.t*self.sigma**2)-1)


class OrnsteinUhlenbeckProcess(StochasticProcessInterface):
    """
    ## Description
    Model describes the evolution of interest rates.
    ### Input:
        - mu (float): Mean of the process
        - sigma (float): Standard deviation of the process
        - alpha (float): Drift scaling parameter
        - n_paths (int): Number of paths to generate
        - n_steps (int): Number of steps
        - T (float): Final time step
        - s_0 (float): Initial value of the stochastic process
    ### LaTeX Formula:
        - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma*dW_{t}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        - Original Source: https://doi.org/10.1103%2FPhysRev.36.823
    """
    def __init__(self, mu: float=0.07, sigma: float=0.1, alpha: float=10.0, n_paths: int=100, n_steps: int=200, T: float=1.0,
                 s_0: float=0.05) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)

    def get_paths(self, analytic_em: bool=False) -> np.ndarray[Any, np.ndarray]:
        """
        ## Description
        Simulates paths of the Ornstein-Uhlenbeck Process using the Euler-Maruyama method.
        This method can be adjusted to use either the standard numerical simulation or an analytic adjustment for Euler-Maruyama.
        ### Input:
            - analytic_em (bool): If True, uses the analytic moments for Euler-Maruyama; otherwise, uses plain Euler-Maruyama simulation
        ### Output:
            - An array (np.ndarray) representing simulated paths of the process for each path and time step
        ### LaTeX Formula:
            - dS_{t} = \\alpha(\\mu - S_{t}) dt + \\sigma dW_{t}
        """
        # Stochastic process
        N = np.random.randn(self.n_steps, self.n_paths)
        s = np.concatenate((self.s_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        # Analytic Euler-Maruyama method
        if analytic_em:
            std = self.sigma*np.sqrt((1-np.exp(-2*self.alpha*self.dt))/(2*self.alpha))
            for i in range(0, self.n_steps):
                s[i+1,:] = self.mu + (s[i,:]-self.mu)*np.exp(-self.alpha*self.dt) + std*N[i,:]
        # Plain Euler-Maruyama method
        else:
            std = self.sigma*np.sqrt(self.dt)
            for i in range(0, self.n_steps):
                s[i+1,:] = s[i,:] + self.alpha*(self.mu-s[i,:])*self.dt + std*N[i,:]
        return s.transpose()

    def get_expectation(self) -> np.ndarray:
        """
        ## Description
        Calculates the expected path of the Ornstein-Uhlenbeck Process, showing the mean-reverting nature of the process over time.
        ### Output:
            - An array (np.ndarray) of expected values of the process at each time step
        ### LaTeX Formula:
            - E[S_t] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
        """
        return self.mu + (self.s_0-self.mu)*np.exp(-self.alpha*self.t)

    def get_variance(self) -> np.ndarray:
        """
        ## Description
        Computes the variance of the Ornstein-Uhlenbeck Process at each time step, providing insights into the variability around the mean.
        ### Output:
            - An array (np.ndarray) of variances of the process at each time step
        ### LaTeX Formula:
            - \\textit{Var}[S_{t}] = \\frac{\\sigma^{2}}{2\\alpha} (1 - e^{-2\\alpha t})
        """
        return (1-np.exp(-2*self.alpha*self.t))*(self.sigma**2)/(2*self.alpha)


class BrownianBridge(StochasticProcessInterface):
    """
    ## Description
    Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
    such as sampling. Also used in scenario generation.
    ### Input:
        - alpha (float): Initial value of the process
        - beta (float): Final value of the process
        - sigma (float): Standard deviation of the process
        - alpha (float): Drift scaling parameter
        - n_paths (int): Number of paths to generate
        - n_steps (int): Number of steps
        - T (float): Final time step
    ### LaTeX Formula:
        - dS_{t} = ((b-a)/(T-t))*dt + \\sigma*dW_{t}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Brownian_bridge
        - Original SOurce: N/A
    """
    def __init__(self, alpha: float=1.0, beta: float=2.0, sigma: float=0.5, n_paths: int=100, n_steps: int=200, T: float=1.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
    
    def get_paths(self) -> np.ndarray[Any, np.ndarray]:
        """
        ## Description
        Generates simulation paths for the Brownian Bridge using the Euler-Maruyama method.\n
        This method approximates the continuous-time process and ensures that the path starts at 'alpha' and ends at 'beta' at time T.
        ### Output:
            - An array (np.ndarray) of simulated paths following the Brownian Bridge for each path and time step
        ### LaTeX Formula:
            - dS_{t} = ((\\beta - \\alpha)/(T - t)) dt + \\sigma dW_{t}
        """
        # Stochastic process
        dW = np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        s = np.concatenate((self.alpha*np.ones((1, self.n_paths)),
                             np.zeros((self.n_steps-1, self.n_paths)), self.beta*np.ones((1, self.n_paths))), axis=0)
        for i in range(0, self.n_steps-1):
            s[i+1,:] = s[i,:] + (self.beta-s[i,:])/(self.n_steps-i+1) +self.sigma*dW[i,:]
        return s.transpose()
    
    def get_expectation(self) -> np.ndarray:
        """
        ## Description
        Calculates the expected path of the Brownian Bridge. It represents the expected value of the process at each time step, starting at 'alpha' and trending towards 'beta'.
        ### Output:
            - An array (np.ndarray) of expected values of the process at each time step
        ### LaTeX Formula:
            - E[S_{t}] = \\alpha + (\\beta - \\alpha) \\frac{t}{T}
        """
        return self.alpha + (self.beta-self.alpha)/self.T*self.t

    def get_variance(self) -> np.ndarray:
        """
        ## Description
        Computes the variance of the Brownian Bridge at each time step. This illustrates how the variability of the process decreases as it approaches the endpoint 'beta' at time T.
        ### Output:
            - An array of variances of the process at each time step
        ### LaTeX Formula:
            - \\text{Var}[S_{t}] = \\frac{t(T-t)}{T} \\sigma^{2}
        """
        return self.t*(self.T-self.t)/self.T


class FellerSquareRootProcess(StochasticProcessInterface):
    """
    ## Description
    Model describes the evolution of interest rates.
    ### Input:
        - mu (float): Mean of the process
        - sigma (float): Standard deviation of the process
        - alpha (float): Drift scaling parameter
        - n_paths (int): Number of paths to generate
        - n_steps (int): Number of steps
        - T (float): Final time step
        - s_0 (float): Initial value of the stochastic process
        - method (FellerSquareRootProcessMethod): Method for computing Feller-Square root process
    ### LaTeX Formula:
        - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma\\sqrt(S_{t})*dW_{t}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
        - Original Source: https://doi.org/10.2307/1911242
    """
    def __init__(self, mu: float=0.05, sigma: float=0.265, alpha: float=5.0, n_paths: int=100, n_steps: int=200, T: float=1.0,
                 s_0: float=0.03, method: FellerSquareRootProcessMethod=FellerSquareRootProcessMethod.EULER_MARUYAMA) -> None:
        # Arguments validation
        type_check(value=method, type_=FellerSquareRootProcessMethod, value_name="method")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
        self.method = method
    
    def get_paths(self) -> np.ndarray[Any, np.ndarray]:
        """
        ## Description
        Simulates paths of the Feller Square-Root Process using different methods: Euler-Maruyama, Analytic Euler-Maruyama, or Exact method, depending on the specified method in the process setup.
        ### Output:
            - An array (np.ndarray) of simulated paths of the process for each path and time step, following the chosen simulation method
        """
        # Stochastic process
        N = np.random.randn(self.n_steps, self.n_paths)
        s = np.concatenate((self.s_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        match self.method:
            case FellerSquareRootProcessMethod.EULER_MARUYAMA:
                for i in range(0, self.n_steps):
                        s[i+1,:] = s[i,:] + self.alpha*(self.mu-s[i,:])*self.dt + self.sigma*np.sqrt(s[i,:]*self.dt)*N[i,:]
                        s[i+1,:] = np.maximum(s[i+1,:], np.zeros((1, self.n_paths)))
            case FellerSquareRootProcessMethod.ANALYTIC_EULER_MARUYAMA:
                a = (self.sigma**2)/self.alpha*(np.exp(-self.alpha*self.dt)-np.exp(-2*self.alpha*self.dt))
                b = self.mu*(self.sigma**2)/(2*self.alpha)*(1-np.exp(-self.alpha*self.dt))**2
                for i in range(0, self.n_steps):
                    s[i+1,:] = self.mu + (s[i,:]-self.mu)*np.exp(-self.alpha*self.dt) + np.sqrt(a*s[i,:]+b)*N[i,:]
                    s[i+1,:] = np.maximum(s[i+1,:], np.zeros((1, self.n_paths)))
            case FellerSquareRootProcessMethod.EXACT:
                d = 4*self.alpha*self.mu/(self.sigma**2)
                k = (self.sigma**2)*(1-np.exp(-self.alpha*self.dt))/(4*self.alpha)
                for i in range(0, self.n_steps):
                    delta = 4*self.alpha*s[i,:]/((self.sigma**2)*(np.exp(self.alpha*self.dt)-1))
                    s[i+1,:] = np.random.noncentral_chisquare(d, delta, (1, self.n_paths))*k
        return s.transpose()
    
    def get_expectation(self) -> np.ndarray:
        """
        ## Description
        Calculates the expected path of the Feller Square-Root Process, showing the mean-reverting nature over time towards the long-term mean \mu.
        ### Output:
            - An array (np.ndarray) of expected values of the process at each time step
        ### LaTeX Formula:
            - E[S_{t}] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
        """
        return self.mu + (self.s_0-self.mu)*np.exp(-self.alpha*self.t)

    def get_variance(self) -> np.ndarray:
        """
        ## Description
        Computes the variance of the Feller Square-Root Process at each time step, providing insights into the variability around the mean.
        ### Output:
            - An array (np.ndarray) of variances of the process at each time step
        """
        return ((self.sigma**2)*(np.exp(-self.alpha*self.t)-np.exp(-self.alpha*2*self.t))*self.s_0/self.alpha +
                (self.sigma**2)*np.exp(-self.alpha*2*self.t)*(np.exp(self.alpha*self.t)-1)**2*self.mu/(2*self.alpha))