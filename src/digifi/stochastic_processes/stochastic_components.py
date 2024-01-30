from typing import Union
import abc
from enum import Enum
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.general_utils import (compare_array_len, type_check, DataClassValidation)



class StationaryTrendType(Enum):
    LINEAR_TREND = 1
    QUADRATIC_TREND = 2
    EXPONENTIAL_TREND = 3



class CustomNoise(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "noise_func") and
                callable(subclass.noise_func))
    
    @abc.abstractmethod
    def noise_func(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Custom noise function.
        """
        raise NotImplementedError



class JumpType(Enum):
    NO_JUMPS = 1
    NORMAL_POISSON_JUMPS = 2
    ASSYMETRIC_BILATERAL_JUMPS = 3
    CUSTOM_JUMPS = 4



@dataclass(slots=True)
class NormalPoissonJumpsParams(DataClassValidation):
    """
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
class AssymetricBilateralJumpsParams(DataClassValidation):
    """
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
    def jump_func(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Custom jump function.
        """
        raise NotImplementedError


class Drift:
    """
    Customizable drift term for a discrete-time stochastic process.
    """
    # TODO: Complete Drift class
    def __init__(self, stationary_trend_type: StationaryTrendType=StationaryTrendType.LINEAR_TREND) -> None:
        type_check(value=stationary_trend_type, type_=StationaryTrendType, value_name="stationary_trend_type")
        self.stationary_trend_type = stationary_trend_type
    
    def base_trend_stationary_term_standard(self, time: float, **kwargs) -> float:
        """
        Predefined trend stationary term for the drift. Returns a trend value for a given stationary trend type.
        - Linear Trend (params: a, b, error_term): a*t + b + error_term
        - Quadratic Trend (params: a, b, c, error_term): a*t**2 + b*t + c + error_term
        - Exponential Trend (params: a, b, error_term): b * exp(a*t) * error_term
        """
        time = float(time)
        match self.stationary_trend_type:
            case StationaryTrendType.LINEAR_TREND:
                a, b = float(kwargs.get("a", 0)), float(kwargs.get("b", 0))
                return a*time + b
            case StationaryTrendType.QUADRATIC_TREND:
                a, b, c = float(kwargs.get("a", 0)), float(kwargs.get("b", 0)), float(kwargs.get("c", 0))
                return a*(time**2) + b*time + c
            case StationaryTrendType.EXPONENTIAL_TREND:
                a, b = float(kwargs.get("a", 0)), float(kwargs.get("b", 0))
                return b*np.exp(a*time)
    
    def get_drift(self, dt: float) -> np.ndarray:
        """
        Get an array of trend values.
        """
        pass



class DifferenceStationary:
    """
    Difference stationary process.
    """
    def __init__(self, n_steps: int, autoregression_parameters: np.ndarray) -> None:
        type_check(value=autoregression_parameters, type_=np.ndarray, value_name="autoregression_parameters")
        self.n_steps = int(n_steps)
        self.autoregression_parameters = autoregression_parameters
    
    def get_autoregression(self, initial_process_values: np.ndarray) -> np.ndarray:
        """
        Get an array of autoregression values.
        """
        compare_array_len(array_1=self.autoregression_parameters, array_2=initial_process_values, array_1_name="autoregression_parameters",
                          array_2_name="initial_process_values")
        process_order = len(self.autoregression_parameters)
        simulated_process_values = initial_process_values.copy()
        for i in range(self.n_steps):
            product = np.dot(self.autoregression_parameters, simulated_process_values[i:i+process_order+1])
            simulated_process_values = np.append(simulated_process_values, product)
        return simulated_process_values[process_order:]



class Diffusion:
    """
    Customizable diffusion term for a discrete-time stochastic process.\n
    Generates a diffusion term:
        Diffusion = sigma * stochastic_factor * noise_func
    """
    def __init__(self, n_steps: int, sigma: float, custom_noise: Union[CustomNoise, None]=None) -> None:
        self.n_steps = int(n_steps)
        self.sigma = float(sigma)
        self.is_custom_noise = False
        if isinstance(custom_noise, CustomNoise):
            self.custom_noise = self.__validate_custom_noise(custom_noise=custom_noise)
        else:
            self.custom_noise = None
    
    def __validate_custom_noise(self, custom_noise: CustomNoise) -> CustomNoise:
        """
        Validate custom noise object to satisfy the computational requirements.
        """
        # Check that custom_noise takes in parameters dt and n_steps
        random_dt = np.random.uniform(low=0, high=1, size=1)[0]
        result = custom_noise.noise_func(n_steps=self.n_steps, dt=random_dt)
        # Check that custom_noise produces an array of length n_steps
        if len(result)!=self.n_steps:
            raise ValueError("The custom noise does not produce an array of length {}.".format(self.n_steps))
        self.is_custom_noise = True
        return custom_noise
    
    def weiner_process(self, dt: float) -> np.ndarray:
        """
        dW_{t} \\approx \\sqrt{dt}*\\mathcal{N}(0, 1)\n
        Generate the Weiner process.\n
        Wikipedia: https://en.wikipedia.org/wiki/Wiener_process\n
        """
        return np.sqrt(float(dt))*np.random.randn(self.n_steps)
    
    def get_diffusion(self, stochastic_factor: float, dt: float) -> np.ndarray:
        """
        Get an array of simulated diffusion values.
        """
        if self.is_custom_noise:
            return self.sigma * float(stochastic_factor) * self.custom_noise.noise_func(n_steps=self.n_steps, dt=dt)
        else:
            return self.sigma * float(stochastic_factor) * self.weiner_process(dt=dt)



class Jump:
    """
    Customizable jump term for a discrete-time stochastic process.
    """
    def __init__(self, n_steps: int, jump_type: JumpType=JumpType.NO_JUMPS,
                 jump_params: Union[NormalPoissonJumpsParams, AssymetricBilateralJumpsParams, None]=None,
                 custom_jump: Union[CustomJump, None]=None) -> None:
        type_check(value=jump_type, type_=JumpType, value_name="jump_type")
        self.n_steps = int(n_steps)
        self.jump_type = jump_type
        self.jump_params = jump_params
        if isinstance(custom_jump, CustomJump):
            self.custom_jump = self.__validate_custom_jump(custom_jump=custom_jump)
        else:
            self.custom_jump = None
    
    def __validate_custom_jump(self, custom_jump: CustomJump) -> CustomJump:
        """
        Validate custom jump object to satisfy the computational requirements.
        """
        # Check that custom_jump takes in parameters dt and n_steps
        random_dt = np.random.uniform(low=0, high=1, size=1)[0]
        result = custom_jump.jump_func(n_steps=self.n_steps, dt=random_dt)
        # Check that custom_jump produces an array of length n_steps
        if len(result)!=self.n_steps:
            raise ValueError("The custom noise does not produce an array of length {}.".format(self.n_steps))
        return custom_jump
    
    def normal_poisson_jumps(self, dt: float, params: NormalPoissonJumpsParams) -> np.ndarray:
        """
        \\text{Jumps Frequency} = Pois(\\lambda dt)\n
        \\{Jumps Distribution} = (\\mu_{j} * \\text{Jumps Frequency}) + (\\sigma_{j} * \\text{Jumps Frequency} * \mathcal{N}(0,1))\n
        """
        # Frequency of jumps
        dP = np.random.poisson(params.lambda_j*float(dt), self.n_steps)
        # Distribution of jumps
        dJ = params.mu_j*dP + params.sigma_j*np.sqrt(dP)*np.random.randn(self.n_steps)
        return dJ
    
    def assymetric_bilateral_jumps(self, dt: float, params: AssymetricBilateralJumpsParams) -> np.ndarray:
        """
        \\text{Jumps Frequency} = Pois(\\lambda dt)\n
        \\text{Assymetric Double Exponential RV} = \mathbb{1}_{p\\leq U(0,1)}*(-\\frac{1}{\\eta_{u}} * ln(\\frac{1-U(0,1)}{p})) +\n
            + \mathbb{1}_{U(0,1)<p}*(\\frac{1}{\\eta_{d}} * ln(\\frac{U(0,1)}{1-p}))\n
        \\text{Jumps Distribution} = (e^{\\text{Assymetric Double Exponential RV}} - 1) * \\text{Jumps Frequency}\n
        """
        # Magnitude of jumps
        dP = np.random.poisson(params.lambda_j*float(dt), self.n_steps)
        # Assymetric double exponential random variable
        u = np.random.uniform(0, 1, self.n_steps)
        y = np.zeros(self.n_steps)
        for i in range(len(u)):
            if params.p<=u[i]:
                y[i] = (-1/params.eta_u) * np.log((1-u[i])/params.p)
            else:
                y[i] = (1/params.eta_d) * np.log(u[i]/(1-params.p))
        # Distribution of jumps
        dJ = (np.exp(y)-1)*dP
        return dJ

    def get_jumps(self, dt: float) -> np.ndarray:
        """
        Get an array of simulated jump values.
        """
        match self.jump_type:
            case JumpType.NO_JUMPS:
                return np.zeros(self.n_steps)
            case JumpType.NORMAL_POISSON_JUMPS:
                type_check(value=self.jump_params, type_=NormalPoissonJumpsParams, value_name="jump_params")
                return self.normal_poisson_jumps(dt=dt, params=self.jump_params)
            case JumpType.ASSYMETRIC_BILATERAL_JUMPS:
                type_check(value=self.jump_params, type_=AssymetricBilateralJumpsParams, value_name="jump_params")
                return self.assymetric_bilateral_jumps(dt=dt, params=self.jump_params)
            case JumpType.CUSTOM_JUMPS:
                return self.custom_jump.jump_func(n_steps=self.n_steps, dt=dt)