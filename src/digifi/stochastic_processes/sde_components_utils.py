from typing import Type
import abc
from enum import Enum
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.general_utils import (type_check, DataClassValidation)



class SDEComponentFunctionType(Enum):
    LINEAR = 1
    QUADRATIC_TIME = 2
    POWER_STOCHASTIC = 3
    CONVERGENCE_TO_VALUE = 4
    REGRESSION_TO_TREND = 5
    CUSTOM = 6



@dataclass(slots=True)
class SDEComponentFunctionParams(DataClassValidation):
    # Linear parameter
    linear_a: float = 1.0
    # Quadratic parameters
    quadratic_time_a: float = 1.0
    quadratic_time_b: float = 1.0
    # Power stochastic parameters
    power_stochastic_a: float = 1.0
    power_stochastic_power: float = 1.0
    # Convergence-to-value parameters
    convergence_to_value_scale: float = 1.0
    convergence_to_value_trend: float = 1.0
    # Regression-to-trend parameters
    regression_to_trend_T: float = 1.0
    regression_to_trend_a: float = 1.0
    regression_to_trend_b: float = 2.0

    def __post_init__(self) -> None:
        super(SDEComponentFunctionParams, self).__post_init__()
        # Validate regression-to-value
        if self.regression_to_trend_b<=self.regression_to_trend_a:
            raise ValueError("The value of the argument regression_to_trend_b should be larger than the value of regression_to_trend_a.")
        if self.regression_to_trend_T<=0:
            raise ValueError("The argument regression_to_trend_T must be positive.")



class SDEComponentFunction:
    """
    Helper functions to define terms for different SDEs.
    """
    @staticmethod
    def linear(a: float, n_paths: int) -> np.ndarray:
        """
        ## Description
        Generates a constant array representing a linear SDE component.

        ### Input:
            - a (float): Constant linear coefficient.
            - n_paths (int): Number of simulation paths.

        ### Output:
            - Array (np.ndarray) representing the linear SDE component for each simulation path.

        ### LaTeX Formula:
            - f(t) = a where a is a constant.
        """

        return float(a) * np.ones(int(n_paths))
    
    @staticmethod
    def quadratic_time(t: float, a: float, b: float, n_paths: int) -> np.ndarray:
        """
        ## Description
        Generates an array representing a quadratic time-dependent SDE component.

        ### Input:
            - t (float): Current time step.
            - a (float): Coefficient for the time-dependent term.
            - b (float): Constant term.
            - n_paths (int): Number of simulation paths.

        ### Output:
            - Array (np.ndarray) representing the quadratic time-dependent SDE component for each simulation path.

        ### LaTeX Formula:
            - f(t) = 2at + b
        """

        return (2*float(a)*float(t) + float(b)) * np.ones(int(n_paths))
    
    @staticmethod
    def power_stochastic(stochastic_values: np.ndarray, a: float, power: float, n_paths: int) -> np.ndarray:
        """
        ## Description
        Generates an array representing a power-law stochastic SDE component.

        ### Input:
            - stochastic_values (np.ndarray): Array of stochastic process values.
            - a (float): Scaling coefficient.
            - power (float): Power-law exponent.
            - n_paths (int): Number of simulation paths.

        ### Output:
            - Array (np.ndarray) representing the power-law stochastic SDE component for each simulation path.

        ### LaTeX Formula:
            - f(X_t) = a \cdot X_t^{power} where X_t represents the stochastic process values.
        """

        type_check(value=stochastic_values, type_=np.ndarray, value_name="stochastic_values")
        if len(stochastic_values)!=int(n_paths):
            raise ValueError("The argument stochastic_values needs to be of the length {} as defined by n_paths.".format(n_paths))
        return float(a) * (stochastic_values**float(power))
    
    @staticmethod
    def convergence_to_value(stochastic_values: np.ndarray, scale: float, trend: float, n_paths: int) -> np.ndarray:
        """
        Returns:
            scale * (trend - stochastic_values)\n
        e.g., Ornstein-Uhlenbeck Process Drift - \\alpha(\\mu - S_{t})
        """
        type_check(value=stochastic_values, type_=np.ndarray, value_name="stochastic_values")
        if len(stochastic_values)!=int(n_paths):
            raise ValueError("The argument stochastic_values needs to be of the length {} as defined by n_paths.".format(n_paths))
        return float(scale) * (float(trend) - stochastic_values)
    
    @staticmethod
    def regression_to_trend(t: float, T: float, a: float, b: float, n_paths: int) -> np.ndarray:
        """
        Returns:
            \\frac{b - a}{T - t}\n
        e.g., Brownian Bridge Drift -\\frac{b - a}{T - t}
        """
        return ((float(b) - float(a)) / (float(T) - float(t))) * np.ones(int(n_paths))



class CustomSDEComponentFunction(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "comp_func") and
                callable(subclass.comp_func))
    
    @abc.abstractmethod
    def comp_func(self, stochastic_values:np.ndarray, t: float, n_paths: int, dt: float) -> np.ndarray:
        """
        Custom SDE component function.
        """
        raise NotImplementedError



def validate_custom_sde_component_function(custom_component: Type[CustomSDEComponentFunction],
                                                  n_paths: int, dt: float) -> Type[CustomSDEComponentFunction]:
    """
    ## Description
    Validates a custom SDE component function ensuring it meets computational requirements.

    ### Input:
        - custom_component (Type[CustomSDEComponentFunction]): Custom component function object.
        - n_paths (int): Number of simulation paths.
        - dt (float): Time step increment.

    ### Output:
        - The validated custom component function object.
    """


    n_paths = int(n_paths)
    dt = float(dt)
    # Check that comp_func takes in parameters stochastic_values, t, n_paths and dt
    stochastic_values = np.ones(n_paths)
    t = float(np.random.uniform(low=0, high=1, size=1)[0])
    result = custom_component.comp_func(stochastic_values=stochastic_values, t=t, n_paths=n_paths, dt=dt)
    # Check that comp_func produces an array of length n_paths
    if len(result)!=n_paths:
        raise ValueError("The custom component function does not produce an array of length {}.".format(n_paths))
    return custom_component



def get_component_values(component_type: SDEComponentFunctionType, component_params: SDEComponentFunctionParams,
                         stochastic_values: np.ndarray, t: float, n_paths: int, dt: float,
                         custom_component_function: Type[CustomSDEComponentFunction]) -> np.ndarray:
    """
    ## Description
    Computes SDE component values for a given component type using the specified parameters and stochastic values.

    ### Input:
        - component_type (SDEComponentFunctionType): Type of the SDE component.
        - component_params (SDEComponentFunctionParams): Parameters for the SDE component.
        - stochastic_values (np.ndarray): Stochastic process values.
        - t (float): Current time step.
        - n_paths (int): Number of simulation paths.
        - dt (float): Time step increment.
        - custom_component_function (Type[CustomSDEComponentFunction]): Custom SDE component function (if applicable).

    ### Output:
        - Array (np.ndarray) of computed SDE component values.

    ### LaTeX Formula:
        - Specific to the chosen SDE component type.

    """

    # Arguments validation
    type_check(value=component_type, type_=SDEComponentFunctionType, value_name="component_type")
    type_check(value=component_params, type_=SDEComponentFunctionParams, value_name="component_params")
    type_check(value=stochastic_values, type_=np.ndarray, value_name="stochastic_values")
    if isinstance(custom_component_function, CustomSDEComponentFunction):
        custom_component_function = validate_custom_sde_component_function(custom_component=custom_component_function, n_paths=n_paths, dt=dt)
    else:
        custom_component_function = None
    t = float(t)
    n_paths = int(n_paths)
    dt = float(dt)
    if len(stochastic_values)!=n_paths:
        raise ValueError("The stochastic values needs to be of the length {} as defined by n_paths.".format(n_paths))
    # Resolve component function
    match component_type:
        case SDEComponentFunctionType.LINEAR:
            return SDEComponentFunction.linear(a=component_params.linear_a, n_paths=n_paths)
        case SDEComponentFunctionType.QUADRATIC_TIME:
            return SDEComponentFunction.quadratic_time(t=t, a=component_params.quadratic_time_a,
                                                                     b=component_params.quadratic_time_b, n_paths=n_paths)
        case SDEComponentFunctionType.POWER_STOCHASTIC:
            return SDEComponentFunction.power_stochastic(stochastic_values=stochastic_values, a=component_params.power_stochastic_a,
                                                                       power=component_params.power_stochastic_power, n_paths=n_paths)
        case SDEComponentFunctionType.CONVERGENCE_TO_VALUE:
            return SDEComponentFunction.convergence_to_value(stochastic_values=stochastic_values,
                                                                           scale=component_params.convergence_to_value_scale,
                                                                           trend=component_params.convergence_to_value_trend,
                                                                           n_paths=n_paths)
        case SDEComponentFunctionType.REGRESSION_TO_TREND:
            return SDEComponentFunction.regression_to_trend(t=t, T=component_params.regression_to_trend_T,
                                                                          a=component_params.regression_to_trend_a,
                                                                          b=component_params.regression_to_trend_b, n_paths=n_paths)
        case SDEComponentFunctionType.CUSTOM:
            return custom_component_function.comp_func(stochastic_values=stochastic_values, t=t, n_paths=n_paths, dt=dt)



class NoiseType(Enum):
    STANDARD_WHITE_NOISE = 1
    WEINER_PROCESS = 2
    CUSTOM_NOISE = 3



class CustomNoise(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "noise_func") and
                callable(subclass.noise_func))
    
    @abc.abstractmethod
    def noise_func(self, n_paths: int, dt: float) -> np.ndarray:
        """
        Custom noise function.
        """
        raise NotImplementedError



class JumpType(Enum):
    NO_JUMPS = 1
    COMPOUND_POISSON_NORMAL = 2
    COMPOUND_POISSON_BILATERAL = 3
    CUSTOM_JUMPS = 4



@dataclass(slots=True)
class CompoundPoissonNormalParams(DataClassValidation):
    """
    ## Description
    Parameters:
        - lambda_j: Jump rate (0 < lambda_j)
        - mu_j: Average jump magnitude
        - sigma_j: Standard deviation of jump magnitude
    """
    lambda_j: float
    mu_j: float
    sigma_j: float

    def validate_lambda_j(self, value: float, **_) -> float:
        if value<=0:
            raise ValueError("The parameter lambda_j must be above zero.")
        return value



@dataclass(slots=True)
class CompoundPoissonBilateralParams(DataClassValidation):
    """
    ## Description
    Parameters:
        - lambda_j: Jump rate (0 < lambda_j)
        - p: Probability of a jump up (0<=p<=1)
        - eta_d: Scaling of jump down
        - eta_u: Scaling of jump up
    """
    lambda_l: float
    p: float
    eta_d: float
    eta_u: float

    def validate_lambda_j(self, value: float, **_) -> float:
        if value<=0:
            raise ValueError("The parameter lambda_j must be above zero.")
        return value
    
    def validate_p(self, value: float, **_) -> float:
        if value<0 or 1<value:
            raise ValueError("The parameter p must be in the range [0,1].")
        return value



class CustomJump(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "jump_func") and
                callable(subclass.jump_func))
    
    @abc.abstractmethod
    def jump_func(self, n_paths: int, dt: float) -> np.ndarray:
        """
        Custom jump function.
        """
        raise NotImplementedError