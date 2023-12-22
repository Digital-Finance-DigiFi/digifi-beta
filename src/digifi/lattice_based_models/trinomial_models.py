from typing import List, Callable, Union
import numpy as np
from src.digifi.lattice_based_models.general import LatticeModelPayoffType



def trinomial_tree_nodes(start_point: float, u: float, d: float, n_steps: int) -> List[np.ndarray]:
    """
    Trinomial tree with the defined parameters presented as an array of layers.
    """
    start_point = float(start_point)
    u = float(u)
    d = float(d)
    s = np.sqrt(u*d)
    if (u<0) or (d<0):
        raise ValueError("The arguments u and d must be positive multiplicative factors of the binomial model.")
    n_steps = int(n_steps)
    trinomial_tree = [np.array([start_point])]
    for layer in range(1, n_steps+1):
        current_layer = s*trinomial_tree[-1]
        u_node = u*trinomial_tree[-1][-1]
        d_node = d*trinomial_tree[-1][0]
        current_layer = np.insert(current_layer, 0, d_node)
        current_layer = np.append(current_layer, u_node)
        trinomial_tree.append(current_layer)
    return trinomial_tree



def trinomial_model(payoff: Callable, start_point: float, u: float, d: float, p_u: float, p_d: float, n_steps: int,
                   payoff_timesteps: Union[List[bool], None]=None) -> float:
    """
    General trinomial model with custom payoff.
    The function assumes that there is a payoff at the final time step.
    This implementation does not discount future cashflows.
    """
    start_point = float(start_point)
    # Movements
    u = float(u)
    d = float(d)
    if (u<0) or (d<0):
        raise ValueError("The arguments u and d must be positive multiplicative factors of the binomial model.")
    # Probabilities
    p_u = float(p_u)
    p_d = float(p_d)
    if ((0<=p_u<=1) is False) or ((0<=p_d<=1) is False):
        raise ValueError("The arguments p_u and p_d must be a defined over a range [0,1].")
    if (p_u+p_d)>1:
        raise ValueError("The probabilities p_u, p_d and (1-p_u-p_d) must add up to 1.")
    p_s = 1-p_u-p_d
    # Steps
    n_steps = int(n_steps)
    if isinstance(payoff_timesteps, type(None)):
        payoff_timesteps = []
        for i in range(n_steps):
            payoff_timesteps.append(True)
    elif isinstance(payoff_timesteps, list):
        if len(payoff_timesteps)!=n_steps:
            raise ValueError("The argument payoff_timesteps should be of length n_steps.")
    else:
        raise TypeError("The argument payoff_timesteps should be a list of boolean values.")
    # Trinomial model
    trinomial_tree = trinomial_tree_nodes(start_point=start_point, u=u, d=d, n_steps=n_steps)
    trinomial_tree[-1] = payoff(trinomial_tree[-1])
    for i in range(len(trinomial_tree)-2, -1, -1):
        layer = np.array([])
        for j in range(1, 2*(i+1)):
            value = p_d*trinomial_tree[i+1][j-1] + p_s*trinomial_tree[i+1][j] + p_u*trinomial_tree[i+1][j+1]
            if payoff_timesteps[i]:
                exercise = payoff(trinomial_tree[i][len(layer)])
                layer = np.append(layer, max(value, exercise))
            else:
                layer = np.append(layer, value)
        trinomial_tree[i] = layer
    return float(trinomial_tree[0][0])



class BrownianMotionTrinomialModel:
    """
    Trinomial models that are scaled to emulate Brownian motion.
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
                self.payoff: Callable = self.__call_payoff
            case LatticeModelPayoffType.PUT:
                self.payoff: Callable = self.__put_payoff
            case _:
                raise ValueError("The argument payoff_type must be of BinomialModelPayoffType type.")
        self.dt = T/n_steps
        if self.dt>=2*(sigma**2)/((r-q)**2):
            raise ValueError("With the given arguments, the condition \Delta t<1\\frac\{\sigma^\{2\}\}\{(r-q)^\{2\}\} is not satisfied.")
        self.u = np.exp(sigma*np.sqrt(2*self.dt))
        self.d = np.exp(-sigma*np.sqrt(2*self.dt))
        self.p_u = ((np.exp((r-q)*self.dt/2)-np.exp(-sigma*np.sqrt(self.dt/2))) / (np.exp(sigma*np.sqrt(self.dt/2))-np.exp(-sigma*np.sqrt(self.dt/2))))**2
        self.p_d = ((np.exp(sigma*np.sqrt(self.dt/2))-np.exp((r-q)*self.dt/2)) / (np.exp(sigma*np.sqrt(self.dt/2))-np.exp(-sigma*np.sqrt(self.dt/2))))**2
    
    def __call_payoff(self, s_t: np.ndarray) -> np.ndarray:
        return np.maximum(s_t-self.k, 0)
    
    def __put_payoff(self, s_t: np.ndarray) -> np.ndarray:
        return np.maximum(self.k-s_t, 0)
    
    def european_option_trinomial_model(self) -> float:
        """
        Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
        """
        payoff_timesteps = []
        for _ in range(self.n_steps):
            payoff_timesteps.append(False)
        return np.exp(-self.r*self.T)*trinomial_model(payoff=self.payoff, start_point=self.s_0, u=self.u, d=self.d, p_u=self.p_u, p_d=self.p_d,
                                                      n_steps=self.n_steps, payoff_timesteps=payoff_timesteps)
    
    def american_option_trinomial_model(self) -> float:
        """
        Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
        """
        payoff_timesteps = []
        for _ in range(self.n_steps):
            payoff_timesteps.append(True)
        return np.exp(-self.r*self.T)*trinomial_model(payoff=self.payoff, start_point=self.s_0, u=self.u, d=self.d, p_u=self.p_u, p_d=self.p_d,
                                                      n_steps=self.n_steps, payoff_timesteps=payoff_timesteps)
    
    def bermudan_option_trinomial_model(self, payoff_timesteps: Union[List[bool], None]=None) -> float:
        """
        Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
        """
        return np.exp(-self.r*self.T)*trinomial_model(payoff=self.payoff, start_point=self.s_0, u=self.u, d=self.d, p_u=self.p_u, p_d=self.p_d,
                                                      n_steps=self.n_steps, payoff_timesteps=payoff_timesteps)
