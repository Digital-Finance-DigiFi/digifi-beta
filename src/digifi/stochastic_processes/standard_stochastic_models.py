import enum
import numpy as np
from src.digifi.stochastic_processes.general import StochasticProcessInterface



class FellerSquareRootProcessMethod(enum.Enum):
    """
    Types of Feller Square-Root Process.
    """
    EULER_MARUYAMA = 1
    ANALYTIC_EULER_MARUYAMA = 2
    EXACT = 3



class ArithmeticBrownianMotion(StochasticProcessInterface):
    """
    dS = mu*dt + sigma*dW
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
        
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the Arithmetic Brownian Motion generated using the Euler-Maruyama method.
        """
        # Stochastic process
        dW = np.sqrt(self.dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = self.mu*self.dt + self.sigma*dW
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=1)
        return np.cumsum(dS, axis=1)
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the Arithmetci Brownian Motion.
        """
        return self.mu*self.t + self.s_0
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Arithmetic Brownian Motion at each time step.
        """
        return self.t*(self.sigma**2)
    
    def get_auto_cov(self, index_t1: int, index_t2: int) -> np.ndarray:
        """
        Auto-covariance of the Arithmetic Brownian Motion between times t1 and t2.
        """
        return (self.sigma**2)*min(self.t[int(index_t1)], self.t[int(index_t2)])     



class GeometricBrownianMotion(StochasticProcessInterface):
    """
    dS = mu*S*dt + sigma*S*dW
    Model describing the evolution of stock prices.
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
    
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the Geometric Brownian Motion generated using the Euler-Maruyama method.
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
        Expected path, E[S], of the Geometric Brownian Motion.
        """
        return self.s_0*np.exp(self.mu*self.t)
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Geometric Brownian Motion at each time step.
        """
        return (self.s_0**2)*np.exp(2*self.mu*self.t)*(np.exp(self.t*self.sigma**2)-1)



class OrnsteinUhlenbeckProcess(StochasticProcessInterface):
    """
    dS = alpha*(mu-S)*dt + sigma*dW
    Model describes the evolution of interest rates.
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
    
    def get_paths(self, analytic_em: bool=False) -> np.ndarray:
        """
        Paths, S, of the Ornsteain_uhlenbeck Process generated using Euler-Maruyama method.
        Intakes an argument analytic_em with bool values. If True, then returns the simulation with the analytic 
        moments for Euler-Maruyama; if False, then returns plain Euler-Maruyama simulation.
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
        Expected path, E[S], of the Ornstein-Uhlenbeck Process.
        """
        return self.mu + (self.s_0-self.mu)*np.exp(-self.alpha*self.t)
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Ornstein-Uhlenbeck Process at each time step.
        """
        return (1-np.exp(-2*self.alpha*self.t))*(self.sigma**2)/(2*self.alpha)
        


class BrownianBridge(StochasticProcessInterface):
    """
    dS = ((b-X)/(T-t))*dt + sigma*dW
    Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
    such as sampling. Also used in scenario generation.
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
    
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the Brownian Bridge generated using the Euler-Maruyama method.
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
        Expected path, E[S], of the Brownian Bridge.
        """
        return self.alpha + (self.beta-self.alpha)/self.T*self.t
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Brownian Bridge.
        """
        return self.t*(self.T-self.t)/self.T



class FellerSquareRootProcess(StochasticProcessInterface):
    """
    dS = alpha*(mu-S)*dt + sigma*sqrt(S)*dW
    Model describes the evolution of interest rates.
    """
    def __init__(self, mu: float=0.05, sigma: float=0.265, alpha: float=5.0, n_paths: int=100, n_steps: int=200, T: float=1.0,
                 s_0: float=0.03) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
    
    def get_paths(self, method: FellerSquareRootProcessMethod=FellerSquareRootProcessMethod.EULER_MARUYAMA) -> np.ndarray:
        """
        Paths, S, of the Feller Square-Root Process generated using either Euler-Maruyama method or the exact method.
        For Euler-Maruyama simulation, set method atribute to FellerSquareRootProcessMethod.EULER_MARUYAMA;
        for Euler-Maruyama with analytic moments, set method atrinute to FellerSquareRootProcessMethod.ANALYTIC_EULER_MARUYAMA;
        for exact solution, set it to FellerSquareRootProcessMethod.EXACT.
        """
        # Stochastic process
        N = np.random.randn(self.n_steps, self.n_paths)
        s = np.concatenate((self.s_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        if method==FellerSquareRootProcessMethod.EULER_MARUYAMA:
            for i in range(0, self.n_steps):
                    s[i+1,:] = s[i,:] + self.alpha*(self.mu-s[i,:])*self.dt + self.sigma*np.sqrt(s[i,:]*self.dt)*N[i,:]
                    s[i+1,:] = np.maximum(s[i+1,:], np.zeros((1, self.n_paths)))
        elif method==FellerSquareRootProcessMethod.ANALYTIC_EULER_MARUYAMA:
            a = (self.sigma**2)/self.alpha*(np.exp(-self.alpha*self.dt)-np.exp(-2*self.alpha*self.dt))
            b = self.mu*(self.sigma**2)/(2*self.alpha)*(1-np.exp(-self.alpha*self.dt))**2
            for i in range(0, self.n_steps):
                s[i+1,:] = self.mu + (s[i,:]-self.mu)*np.exp(-self.alpha*self.dt) + np.sqrt(a*s[i,:]+b)*N[i,:]
                s[i+1,:] = np.maximum(s[i+1,:], np.zeros((1, self.n_paths)))
        elif method==FellerSquareRootProcessMethod.EXACT:
            d = 4*self.alpha*self.mu/(self.sigma**2)
            k = (self.sigma**2)*(1-np.exp(-self.alpha*self.dt))/(4*self.alpha)
            for i in range(0, self.n_steps):
                delta = 4*self.alpha*s[i,:]/((self.sigma**2)*(np.exp(self.alpha*self.dt)-1))
                s[i+1,:] = np.random.noncentral_chisquare(d, delta, (1, self.n_paths))*k
        else:
            raise ValueError("Provided method is not recognized. Use FellerSquareRootProcessMethod enum to select a supported method.")
        return s.transpose()
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the Feller Square-Root Process.
        """
        return self.mu + (self.s_0-self.mu)*np.exp(-self.alpha*self.t)
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Feller Square-Root Process.
        """
        return ((self.sigma**2)*(np.exp(-self.alpha*self.t)-np.exp(-self.alpha*2*self.t))*self.s_0/self.alpha +
                (self.sigma**2)*np.exp(-self.alpha*2*self.t)*(np.exp(self.alpha*self.t)-1)**2*self.mu/(2*self.alpha))