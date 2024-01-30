from typing import List, Callable, Union
import numpy as np
from src.digifi.utilities.maths_utils import n_choose_r
from src.digifi.lattice_based_models.general import (LatticeModelPayoffType, LatticeModelInterface)



def binomial_tree_nodes(start_point: float, u: float, d: float, n_steps: int) -> List[np.ndarray]:
    """
    Binomial tree with the defined parameters presented as an array of layers.
    """
    start_point = float(start_point)
    u = float(u)
    d = float(d)
    if (u<0) or (d<0):
        raise ValueError("The arguments u and d must be positive multiplicative factors of the binomial model.")
    n_steps = int(n_steps)
    binomial_tree = []
    for layer in range(n_steps+1):
        current_layer = np.array([])
        for i in range(layer+1):
            node = start_point * (u**i) * (d**(layer-i))
            current_layer = np.append(current_layer, node)
        binomial_tree.append(current_layer)
    return binomial_tree



def binomial_model(payoff: Callable, start_point: float, u: float, d: float, p_u: float, n_steps: int,
                   payoff_timesteps: Union[List[bool], None]=None) -> float:
    """
    General binomial model with custom payoff.\n
    The function assumes that there is a payoff at the final time step.\n
    This function does not discount future cashflows.\n
    """
    start_point = float(start_point)
    u = float(u)
    d = float(d)
    if (u<0) or (d<0):
        raise ValueError("The arguments u and d must be positive multiplicative factors of the binomial model.")
    p_u = float(p_u)
    if (0<=p_u<=1) is False:
        raise ValueError("The argument p_u must be a defined over a range [0,1].")
    n_steps = int(n_steps)
    if isinstance(payoff_timesteps, type(None)):
        payoff_timesteps = []
        for i in range(n_steps):
            payoff_timesteps.append(True)
    elif isinstance(payoff_timesteps, list):
        if len(payoff_timesteps)!=n_steps:
            raise ValueError("The argument payoff_timesteps should be of length n_steps.")
        if all(isinstance(value, bool) for value in payoff_timesteps) is False:
            raise TypeError("The argument payoff_timesteps should be a list of boolean values.")
    else:
        raise TypeError("The argument payoff_timesteps should be a list of boolean values.")
    # Binomial model
    binomial_tree = []
    layer = np.array([])
    # Final layer
    for i in range(n_steps+1):
        value = payoff(start_point * (u**i) * (d**(n_steps-i)))
        layer = np.append(layer, value)
    binomial_tree.append(layer)
    # Layers before the final layer
    for j in range(n_steps-1, -1, -1):
        layer = np.array([])
        for i in range(0, j+1):
            value = p_u*binomial_tree[-1][i+1] + (1-p_u)*binomial_tree[-1][i]
            if payoff_timesteps[j]:
                exercise = payoff(s_t=start_point * (u**i) * (d**(j-i)))
                layer = np.append(layer, max(value, exercise))
            else:
                layer = np.append(layer, value)
        binomial_tree.append(layer)
    return float(binomial_tree[-1][0])



class BrownianMotionBinomialModel(LatticeModelInterface):
    """
    Binomial models that are scaled to emulate Brownian motion.
    """
    def __init__(self, s_0: float, k: float, T: float, r: float, sigma: float, q: float, n_steps: int,
                 payoff_type: LatticeModelPayoffType=LatticeModelPayoffType.CALL) -> None:
        self.s_0 = float(s_0)
        self.k = float(k)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.n_steps = int(n_steps)
        match payoff_type:
            case LatticeModelPayoffType.CALL:
                self.payoff: Callable = self.call_payoff
            case LatticeModelPayoffType.PUT:
                self.payoff: Callable = self.put_payoff
            case _:
                raise ValueError("The argument payoff_type must be of BinomialModelPayoffType type.")
        self.dt = T/n_steps
        self.u = np.exp(sigma*np.sqrt(self.dt))
        self.d = np.exp(-sigma*np.sqrt(self.dt))
    
    def call_payoff(self, s_t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return max(s_t-self.k, 0)
    
    def put_payoff(self, s_t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return max(self.k-s_t, 0)
    
    def european_option(self) -> float:
        """
        Binomial model that computes the payoffs for each path and computes the weighted average of paths based on probability.
        """
        p = (np.exp((self.r-self.q)*self.dt) - self.d)/(self.u-self.d)
        value = 0
        for i in range(self.n_steps+1):
            node_probability = n_choose_r(self.n_steps, i) * (p**i) * ((1-p)**(self.n_steps-i))
            s_T = self.s_0 * (self.u**i) * (self.d**(self.n_steps-i))
            value += self.payoff(s_t=s_T)*node_probability
        return float(value*np.exp(-self.r*self.T))
    
    def american_option(self) -> float:
        """
        Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
        """
        payoff_timesteps = []
        for _ in range(self.n_steps):
            payoff_timesteps.append(True)
        return self.bermudan_option(payoff_timesteps=payoff_timesteps)
    
    def bermudan_option(self, payoff_timesteps: Union[List[bool], None]=None) -> float:
        """
        Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
        """
        if isinstance(payoff_timesteps, type(None)):
            payoff_timesteps = []
            for i in range(self.n_steps):
                payoff_timesteps.append(True)
        elif isinstance(payoff_timesteps, list):
            if len(payoff_timesteps)!=self.n_steps:
                raise ValueError("The argument payoff_timesteps should be of length n_steps.")
            if all(isinstance(value, bool) for value in payoff_timesteps) is False:
                raise TypeError("The argument payoff_timesteps should be a list of boolean values.")
        else:
            raise TypeError("The argument payoff_timesteps should be a list of boolean values.")
        p_u = (np.exp(-self.q*self.dt) - np.exp(-self.r*self.dt)*self.d)/(self.u-self.d)
        p_d = np.exp(-self.r*self.dt) - p_u
        binomial_tree = []
        layer = np.array([])
        for i in range(self.n_steps+1):
            layer = np.append(layer, self.payoff(s_t=self.s_0 * (self.u**i) * (self.d**(self.n_steps-i))))
        binomial_tree.append(layer)
        for j in range(self.n_steps-1, -1, -1):
            layer = np.array([])
            for i in range(0, j+1):
                value = p_u*binomial_tree[-1][i+1] + p_d*binomial_tree[-1][i]
                if payoff_timesteps[-j]:
                    exercise = self.payoff(s_t=self.s_0 * (self.u**i) * (self.d**(j-i)))
                    layer = np.append(layer, max(value, exercise))
                else:
                    layer = np.append(layer, value)
            binomial_tree.append(layer)
        return float(binomial_tree[-1][0])