from typing import Union, Tuple
import numpy as np
import plotly.graph_objects as go
from src.digifi.stochastic_processes.general import StochasticProcessInterface
from src.digifi.plots.stochastic_models_plots import plot_stochastic_paths



class ConstantElasticityOfVariance(StochasticProcessInterface):
    """
    dS = mu*S*dt + sigma*S^(beta+1)*dW
    Model used to reproduce the volatility smile effect.
    """
    def __init__(self, mu: float=0.2, sigma: float=0.4, beta: float=0.0, n_paths: int=100, n_steps: int=200, T: float=1.0,
                 s_0: float=100.0) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
    
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the Constant Elasticity of Variance Process.
        """
        # Stochastic process
        dW = np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        s = np.concatenate((self.s_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0) 
        for i in range(0, self.n_steps):
            s[i+1,:] = s[i,:] + self.mu*s[i,:]*self.dt + self.sigma*(s[i,:]**(self.beta+1))*dW[i,:]
            s[i+1,:] = np.maximum(s[i+1,:], np.zeros((1, self.n_paths)))
        return s
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected paths, E[S], of the Constant Elasticity of Variance Process.
        """
        return self.s_0*np.exp(self.mu*self.t)
    
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Plot of the random paths taken by the Constant Elasticity of Variance Process.
        """
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths(), expected_path=expected_path, return_fig_object=return_fig_object)



class HestonStochasticVolatility(StochasticProcessInterface):
    """"
    dS = mu*S*dt + sqrt(v)*S*dW_{S}
    dv = k*(theta-v)*dt + epsilon*sqrt(v)dW_{v}
    Model describes the evolution of stock price and its volatility.
    """
    def __init__(self, mu: float=0.1, k: float=5.0, theta: float=0.07, epsilon: float=0.2, rho: float=0.0, n_paths: int=100,
                 n_steps: int=200, T: float=1.0, s_0: float=100.0, v_0: float=0.03) -> None:
        self.mu = float(mu)
        self.k = float(k)
        self.theta = float(theta)
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
        self.v_0 = float(v_0)
    
    def get_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Paths, S, of the Heston Stochastic Volatility Process.
        """
        Nv = np.random.randn(self.n_steps, self.n_paths)
        N = np.random.randn(self.n_steps, self.n_paths)
        NS = self.rho*Nv + np.sqrt(1-self.rho**2)*N
        v = np.concatenate((self.v_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        s = np.concatenate((self.s_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        a = (self.epsilon**2)/self.k*(np.exp(-self.k*self.dt)-np.exp(-2*self.k*self.dt))
        b = self.theta*(self.epsilon**2)/(2*self.k)*(1-np.exp(-self.k*self.dt))**2
        # Stochastic volatility
        for i in range(0, self.n_steps):
            v[i+1,:] = self.theta + (v[i,:]-self.theta)*np.exp(-self.k*self.dt) + np.sqrt(a*v[i,:]+b)*Nv[i,:]
            v[i+1,:] = np.maximum(v[i+1,:], np.zeros((1, self.n_paths)))
        # Stochastic process
        for j in range(0, self.n_steps):
            s[j+1,:] = s[j,:] + (self.mu-0.5*v[j,:])*self.dt + self.epsilon*np.sqrt(v[j,:]*self.dt)*NS[j,:]
            s[j+1,:] = np.maximum(s[j+1,:], np.zeros((1, self.n_paths)))
        return (s, v)
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected paths, E[S], of the Heston Stochastic Volatility Process.
        """
        return (self.s_0 + (self.mu-0.5*self.theta)*self.t
                + (self.theta-self.v_0)*(1-np.exp(-self.k*self.t))/(2*self.k))
    
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Plot of the random paths taken by the Heston Stochastic Volatility Process.
        """
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths()[0], expected_path=expected_path, return_fig_object=return_fig_object)



class VarianceGammaProcess(StochasticProcessInterface):
    """
    dS = mu*dG(t) + sigma*dW(dG(t))
    Model used in option pricing.
    """
    def __init__(self, mu: float=0.2, sigma: float=0.3, rate: float=20.0, n_paths: int=100, n_steps: int=200, T: float=1.0, s_0: float=0.03) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.kappa = float(1/rate)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
        self.rate = float(rate)
    
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the Variance Gamma Process.
        """
        # Stochastic process
        dG = np.random.gamma(self.dt/self.kappa, self.kappa, (self.n_steps, self.n_paths))
        dS = self.mu*dG + self.sigma*np.random.randn(self.n_steps, self.n_paths)*np.sqrt(dG)
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=0)
        return np.cumsum(dS, axis=0)
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the Variance Gamma Process.
        """
        return self.mu*self.t + self.s_0 
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Variance Gamma Process.
        """
        return (self.sigma**2 + (self.mu**2)/self.rate)*self.t
    
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Plot of the random paths taken by the Variance Gamma Process.
        """
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths(), expected_path=expected_path, return_fig_object=return_fig_object)