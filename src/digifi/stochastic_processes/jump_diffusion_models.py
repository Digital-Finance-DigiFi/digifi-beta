from typing import Union
import numpy as np
import plotly.graph_objects as go
from src.digifi.stochastic_processes.general import StochasticProcessInterface
from src.digifi.plots.stochastic_models_plots import plot_stochastic_paths



class MertonJumpDiffusionProcess(StochasticProcessInterface):
    """S = (mu-0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i
    Model describes stock price with continuous movement that have rare large jumps."""
    def __init__(self, mu_s: float, sigma_s: float, mu_j: float, sigma_j: float, lambda_j: float, n_paths: int, n_steps: int, T: float,
                 s_0: float) -> None:
        self.mu_s = float(mu_s)
        self.sigma_s = float(sigma_s)
        self.mu_j = float(mu_j)
        self.sigma_j = float(sigma_j)
        self.lambda_j = float(lambda_j)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
    
    def get_paths(self) -> np.ndarray:
        """Paths, S, of the Merton Jump-Diffusion Process."""
        # Stochastic process
        dX = (self.mu_s-0.5*self.sigma_s**2)*self.dt + self.sigma_s*np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambda_j*self.dt, (self.n_steps, self.n_paths))
        # Jump process
        dJ = self.mu_j*dP + self.sigma_j*np.sqrt(dP)*np.random.randn(self.n_steps, self.n_paths)
        dS = dX + dJ
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=0)
        return np.cumsum(dS, axis=0)
    
    def get_expectation(self) -> np.ndarray:
        """Expected path, E[S], of the Merton Jump-Diffusion Process."""
        return (self.mu_s+self.lambda_j*self.mu_j)*self.t+self.s_0
    
    def get_variance(self) -> np.ndarray:
        """Variance, Var[S], of the Merton Jump-Diffusion Process."""
        return (self.mu_s**2+self.lambda_j*(self.mu_j**2+self.sigma_j**2))*self.t
    
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[np.ndarray, None]:
        """Plot of the random paths taken by the Merton Jump-Diffusion Process."""
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths(), expected_path=expected_path, return_fig_object=return_fig_object)



class KouJumpDiffusionProcess(StochasticProcessInterface):
    """S = mu*t +sigma*W(t) + sum_{i=1}^{N(t)} Z_i
    Model describes stock price with continuous movement that have rare large jumps, with the jump sizes following a double 
    exponential distribution."""
    def __init__(self, mu: float, sigma: float, lambda_n: float, eta_1: float, eta_2: float, p: float, n_paths: int, n_steps: int,
                 T: float, s_0: float) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lambda_n = float(lambda_n)
        self.eta_1 = float(eta_1)
        self.eta_2 = float(eta_2)
        self.p = float(p)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/n_steps
        self.t = np.arange(0, T+self.dt, self.dt)
        self.s_0 = float(s_0)
    
    def get_paths(self) -> np.ndarray:
        """Returns the paths, S, for the Kou Jump-Diffusion Process"""
        # Stochstic process
        dX = (self.mu-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambda_n*self.dt, (self.n_steps, self.n_paths))
        #Bilateral exponential random variable       
        u = np.random.uniform(0,1, (self.n_steps, self.n_paths))
        z = np.zeros((self.n_steps, self.n_paths))
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                if u[j,i]>=self.p:
                    z[j,i]=(-1/self.eta_1)*np.log((1-u[j,i])/self.p)
                elif u[j,i]<self.p:
                    z[j,i]=(1/self.eta_2)*np.log(u[j,i]/(1-self.p))
        dJ = (np.exp(z)-1)*dP
        dS = dX + dJ
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=0)
        return np.cumsum(dS, axis=0)
    
    def get_expectation(self) -> np.ndarray:
        """Expected path, E[S], of the Kou Jump-Diffusion Process."""
        return (self.mu + self.lambda_n*(self.p/self.eta_1-(1-self.p)/self.eta_2))*self.t + self.s_0
    
    def get_variance(self) -> np.ndarray:
        """Variance, Var[S], of the Kou Jump-Diffusion Process."""
        return (self.sigma**2 + 2*self.lambda_n*(self.p/(self.eta_1**2)+(1-self.p)/(self.eta_2**2)))*self.t
    
    def plot(self, plot_expected: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """Plot of the random paths taken by the Kou Jump-Diffusion Process."""
        expected_path = None
        if plot_expected:
            expected_path = self.get_expectation()
        return plot_stochastic_paths(paths=self.get_paths(), expected_path=expected_path, return_fig_object=return_fig_object)