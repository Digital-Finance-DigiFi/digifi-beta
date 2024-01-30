import numpy as np
from src.digifi.stochastic_processes.general import StochasticProcessInterface



class MertonJumpDiffusionProcess(StochasticProcessInterface):
    """
    S_{t} = (\\mu-0.5*\\sigma^2)*t + \\sigma*W_{t} + sum_{i=1}^{N(t)} Z_{i}\n
    Model describes stock price with continuous movement that have rare large jumps.\n
    Wikipedia: https://en.wikipedia.org/wiki/Jump_diffusion#:~:text=a%20restricted%20volume-,In%20economics%20and%20finance,-%5Bedit%5D\n
    """
    def __init__(self, mu_s: float=0.2, sigma_s: float=0.3, mu_j: float=-0.1, sigma_j: float=0.15, lambda_j: float=0.5, n_paths: int=100,
                 n_steps: int=200, T: float=1.0, s_0: float=100.0) -> None:
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
        """
        Paths, S, of the Merton Jump-Diffusion Process.
        """
        # Stochastic process
        dX = (self.mu_s-0.5*self.sigma_s**2)*self.dt + self.sigma_s*np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambda_j*self.dt, (self.n_steps, self.n_paths))
        # Jump process
        dJ = self.mu_j*dP + self.sigma_j*np.sqrt(dP)*np.random.randn(self.n_steps, self.n_paths)
        dS = dX + dJ
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=0)
        return np.cumsum(dS, axis=0).transpose()
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the Merton Jump-Diffusion Process.
        """
        return (self.mu_s+self.lambda_j*self.mu_j)*self.t+self.s_0
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Merton Jump-Diffusion Process.
        """
        return (self.mu_s**2+self.lambda_j*(self.mu_j**2+self.sigma_j**2))*self.t



class KouJumpDiffusionProcess(StochasticProcessInterface):
    """
    dS_{t} = \\mu*dt + \\sigma*dW_{t} + d(sum_{i=1}^{N(t)}(V_{i}-1))\n
    where V_{i} is i.i.d. non-negative random variables such that Y = log(V) is the assymetric double exponential distribution with density:\n
    f_{Y}(y) = p*\\eta_{1}*e^{-\\eta_{1}y}\mathbb{1}_{0\\leq0} + (1-p)*\\eta_{2}*e^{\\eta_{2}y}\mathbb{y<0}
    Model describes stock price with continuous movement that have rare large jumps, with the jump sizes following a double 
    exponential distribution.\n
    Wikipedia: N/A\n
    Original Source: https://www.columbia.edu/~sk75/MagSci02.pdf
    """
    def __init__(self, mu: float=0.2, sigma: float=0.3, lambda_n: float=0.5, eta_1: float=9.0, eta_2: float=5.0, p: float=0.5,
                 n_paths: int=100, n_steps: int=200, T: float=1.0, s_0: float=100.0) -> None:
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
        """
        Paths, S, of the Kou Jump-Diffusion Process.
        """
        # Stochstic process
        dX = (self.mu-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambda_n*self.dt, (self.n_steps, self.n_paths))
        # Assymetric double exponential random variable
        u = np.random.uniform(0,1, (self.n_steps, self.n_paths))
        y = np.zeros((self.n_steps, self.n_paths))
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                if u[j,i]>=self.p:
                    y[j,i]=(-1/self.eta_1)*np.log((1-u[j,i])/self.p)
                elif u[j,i]<self.p:
                    y[j,i]=(1/self.eta_2)*np.log(u[j,i]/(1-self.p))
        dJ = (np.exp(y)-1)*dP
        dS = dX + dJ
        # Data formatting
        dS = np.insert(dS, 0, self.s_0, axis=0)
        return np.cumsum(dS, axis=0).transpose()
    
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the Kou Jump-Diffusion Process.
        """
        return (self.mu + self.lambda_n*(self.p/self.eta_1-(1-self.p)/self.eta_2))*self.t + self.s_0
    
    def get_variance(self) -> np.ndarray:
        """
        Variance, Var[S], of the Kou Jump-Diffusion Process.
        """
        return (self.sigma**2 + 2*self.lambda_n*(self.p/(self.eta_1**2)+(1-self.p)/(self.eta_2**2)))*self.t