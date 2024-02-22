from typing import (List, Union, Type)
import numpy as np
from digifi.utilities.general_utils import type_check
from digifi.utilities.maths_utils import n_choose_r
from digifi.lattice_based_models.general import (LatticeModelPayoffType, LatticeModelInterface)
from digifi.financial_instruments.derivatives_utils import (CustomPayoff, LongCallPayoff, LongPutPayoff, validate_custom_payoff)



def binomial_tree_nodes(start_point: float, u: float, d: float, n_steps: int) -> List[np.ndarray]:
    """
    Binomial tree with the defined parameters presented as an array of layers.
    ### Input:
        - start_point (float): Starting value
        - u (float): Upward movement factor, must be positive
        - d (float): Downward movement factor, must be positive
        - n_steps (int): Number of steps in the tree
    ### Output:
        - List of layers (List[np.ndarrays]) with node values at each step
    ### Exceptions:
        - ValueError if 'u' or 'd' is non-positive
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



def binomial_model(payoff: Type[CustomPayoff], start_point: float, u: float, d: float, p_u: float, n_steps: int,
                   payoff_timesteps: Union[List[bool], None]=None) -> float:
    """
    ## Description
    General binomial model with custom payoff. It constructs a binomial tree with given parameters; calculates the value at each node considering the custom payoff and probability; 
    and aggregates these values to determine the final fair value of the option.\n
    Note: The function assumes that there is a payoff at the final time step.\n
    Note: This function does not discount future cashflows.
    ### Input Extensions:
        - payoff (Type[CustomPayoff]): Custom payoff object defining the payoff at each node
        - p_u (float): Probability of an upward movement, must be in [0,1]
        - payoff_timesteps (Union[List[bool], None]): A list indicating whether there's a payoff at each timestep. Defaults to payoff at every step if None
    ### Output:
        - The fair value (float) calculated by the binomial model
    ### Additional Exceptions:
        - ValueError if 'p_u' is not in [0,1]
        - ValueError if 'payoff_timesteps' length does not match 'n_steps'
        - TypeError if 'payoff_timesteps' is not a list of boolean values
    """
    payoff = validate_custom_payoff(custom_payoff=payoff)
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
        value = payoff.payoff(s_t=np.array([start_point * (u**i) * (d**(n_steps-i))]))[0]
        layer = np.append(layer, value)
    binomial_tree.append(layer)
    # Layers before the final layer
    for j in range(n_steps-1, -1, -1):
        layer = np.array([])
        for i in range(0, j+1):
            value = p_u*binomial_tree[-1][i+1] + (1-p_u)*binomial_tree[-1][i]
            if payoff_timesteps[j]:
                exercise = payoff.payoff(s_t=np.array([start_point * (u**i) * (d**(j-i))]))[0]
                layer = np.append(layer, max(value, exercise))
            else:
                layer = np.append(layer, value)
        binomial_tree.append(layer)
    return float(binomial_tree[-1][0])



class BrownianMotionBinomialModel(LatticeModelInterface):
    """
    ## Description
    Binomial models that are scaled to emulate Brownian motion. This model uses a binomial lattice
    approach to approximate the continuous path of Brownian motion, specifically for option pricing.\n
    This model calculates the up (u) and down (d) factors using the volatility and time step (dt), ensuring the binomial model aligns with the log-normal distribution of stock prices in the Black-Scholes model. 
    Depending on the payoff type, it sets the appropriate payoff function. For more detailed theory, refer to the Cox-Ross-Rubinstein model and its alignment with the Black-Scholes model in financial literature.\n
    This technique is rooted in the Cox-Ross-Rubinstein (CRR) model, adapting it to mirror the properties of Brownian motion.
    ### Input:
        - s_0 (float): Initial stock price
        - k (float): Strike price
        - T (float): Time to maturity (in years)
        - r (float): Risk-free interest rate (annual)
        - sigma (float): Volatility of the underlying asset
        - q (float): Dividend yield
        - n_steps (int): Number of steps in the binomial model
        - payoff_type (LatticeModelPayoffType): Type of the option payoff (e.g., long call, long put)
        - custom_payoff (Union[Type[CustomPayoff], None]): Custom payoff function, if applicable
    ### Exceptions:
        - ValueError if the `custom_payoff` is required but not provided or is invalid
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Binomial_options_pricing_model
        - Original Source: N/A
    """
    def __init__(self, s_0: float, k: float, T: float, r: float, sigma: float, q: float, n_steps: int,
                 payoff_type: LatticeModelPayoffType=LatticeModelPayoffType.LONG_CALL,
                 custom_payoff: Union[Type[CustomPayoff], None]=None) -> None:
        # Arguments validation
        type_check(value=payoff_type, type_=LatticeModelPayoffType, value_name="payoff_type")
        # BrownianMotionBinomialModel class parameters
        self.s_0 = float(s_0)
        self.k = float(k)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.n_steps = int(n_steps)
        match payoff_type:
            case LatticeModelPayoffType.LONG_CALL:
                self.payoff = self.__long_call_payoff().payoff
            case LatticeModelPayoffType.LONG_PUT:
                self.payoff = self.__long_put_payoff().payoff
            case LatticeModelPayoffType.CUSTOM:
                if isinstance(custom_payoff, CustomPayoff):
                    self.payoff = validate_custom_payoff(custom_payoff=custom_payoff).payoff
                else:
                    raise ValueError("For the CUSTOM payoff type, the argument custom_payoff must be defined.")
        self.dt = T/n_steps
        self.u = np.exp(sigma*np.sqrt(self.dt))
        self.d = np.exp(-sigma*np.sqrt(self.dt))
    
    def __long_call_payoff(self) -> Type[CustomPayoff]:
        return LongCallPayoff(k=self.k)
    
    def __long_put_payoff(self) -> Type[CustomPayoff]:
        return LongPutPayoff(k=self.k)
    
    def european_option(self) -> float:
        """
        ## Description
        Binomial model that computes the payoffs for each path and computes the weighted average of paths based on probability.
        ### Output:
            - The present value of the European option (float)
        """
        p = (np.exp((self.r-self.q)*self.dt) - self.d)/(self.u-self.d)
        value = 0
        for i in range(self.n_steps+1):
            node_probability = n_choose_r(self.n_steps, i) * (p**i) * ((1-p)**(self.n_steps-i))
            s_T = self.s_0 * (self.u**i) * (self.d**(self.n_steps-i))
            value += self.payoff(s_t=np.array([s_T]))[0]*node_probability
        return float(value*np.exp(-self.r*self.T))
    
    def american_option(self) -> float:
        """
        ## Description
        Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
        ### Output:
            - The present value of the American option (float)
        """
        payoff_timesteps = []
        for _ in range(self.n_steps):
            payoff_timesteps.append(True)
        return self.bermudan_option(payoff_timesteps=payoff_timesteps)
    
    def bermudan_option(self, payoff_timesteps: Union[List[bool], None]=None) -> float:
        """
        ## Description
        Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
        ### Input:
            - payoff_timesteps (Union[List[bool], None]): Indicators for exercise opportunity at each timestep
        ### Output:
            - The present value of the Bermudan option (float)
        ### Exceptions:
            - ValueError if 'payoff_timesteps' length does not match 'n_steps'
            - TypeError if 'payoff_timesteps' is not a list of boolean values
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
            layer = np.append(layer, self.payoff(s_t=np.array([self.s_0 * (self.u**i) * (self.d**(self.n_steps-i))]))[0])
        binomial_tree.append(layer)
        for j in range(self.n_steps-1, -1, -1):
            layer = np.array([])
            for i in range(0, j+1):
                value = p_u*binomial_tree[-1][i+1] + p_d*binomial_tree[-1][i]
                if payoff_timesteps[-j]:
                    exercise = self.payoff(s_t=np.array([self.s_0 * (self.u**i) * (self.d**(j-i))]))[0]
                    layer = np.append(layer, max(value, exercise))
                else:
                    layer = np.append(layer, value)
            binomial_tree.append(layer)
        return float(binomial_tree[-1][0])