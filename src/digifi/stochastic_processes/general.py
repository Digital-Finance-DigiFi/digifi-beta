from typing import Union
import abc
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import compare_array_len, type_check
from src.digifi.stochastic_processes.stochastic_components import (CustomNoise, JumpType, NormalPoissonJumpsParams, AssymetricBilateralJumpsParams,
                                                                   CustomJump, Drift, DifferenceStationary, Diffusion, Jump)



class StochasticProcessType(Enum):
    TREND_STATIONARY = 1
    DIFFERENCE_STATIONARY = 2



class StochasticProcessInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "get_paths") and
                callable(subclass.get_paths) and
                hasattr(subclass, "get_expectation") and
                callable(subclass.get_expectation))
    
    @abc.abstractmethod
    def get_paths(self) -> np.ndarray:
        """
        Paths, S, of the stochastic process.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_expectation(self) -> np.ndarray:
        """
        Expected path, E[S], of the stochastic process.
        """
        raise NotImplementedError



class StochasticProcess:
    """
    Defines a stochastic process of form:
        dS_{t} = drift | difference stationary process + diffusion
    Parameters:
        - T: Final time step
        - s_0: Starting point for the process
        - n_steps: Number of steps a process takes between start and T
        - n_paths: Number of paths (i.e., iterations) to return
        - process_type: Is process trend stationary or difference stationary
        - autoregression_parameters: Constants used for the autoregressive terms in the difference stationary process
        - initial_process_values: Initial values of the autoregressive terms in the difference stationary process
        - diffusion_constant: Constatnt value by which the diffusion is multiplied (i.e., sigma)
        - custom_noise: Custom noise object used to generate noise inside diffusion
    """
    def __init__(self, T: float, s_0: float, n_steps: int, n_paths: int=100, process_type: StochasticProcessType=StochasticProcessType.TREND_STATIONARY,
                 # Difference stationary
                 autoregression_parameters: np.ndarray=np.array([]), initial_process_values: np.ndarray=np.array([]),
                 # Diffusion
                 diffusion_contant: float=1, custom_noise: Union[CustomNoise, None]=None,
                 #Jumps
                 jump_type: JumpType=JumpType.NO_JUMPS, jump_params: Union[NormalPoissonJumpsParams, AssymetricBilateralJumpsParams, None]=None,
                 custom_jump: Union[CustomJump, None]=None) -> None:
        # Arguments validation
        type_check(value=process_type, type_=StochasticProcessType, value_name="process_type")
        compare_array_len(array_1=autoregression_parameters, array_2=initial_process_values, array_1_name="autoregression_parameters",
                          array_2_name="initial_process_values")
        # General parameters
        self.T = float(T)
        self.s_0 = float(s_0)
        self.n_steps = int(n_steps)
        self.dt = self.T/self.n_steps
        self.n_paths = int(n_paths)
        self.process_type = process_type
        # Drift term and parameters
        self.drift = self.construct_drift()
        # Difference stationary term and parameters
        self.initial_process_values = initial_process_values
        self.difference_stationary = self.construct_difference_stationary(autoregression_parameters=autoregression_parameters)
        # Diffusion term and parameters
        self.diffusion = self.construct_diffusion(sigma=float(diffusion_contant), custom_noise=custom_noise)
        # Jump term and parameters
        self.jump = self.construct_jump(jump_type=jump_type, jump_params=jump_params, custom_jump=custom_jump)
    
    def simulate_path(self) -> np.ndarray:
        match self.process_type:
            case StochasticProcessType.TREND_STATIONARY:
                stochastic_trend = self.drift.get_drift(dt=self.dt)
            case StochasticProcessType.DIFFERENCE_STATIONARY:
                stochastic_trend =  self.difference_stationary.get_autoregression(initial_process_values=self.initial_process_values)
        return stochastic_trend + self.diffusion.get_diffusion(dt=self.dt) + self.jump.get_jumps(dt=self.dt)
    
    def construct_drift(self) -> Drift:
        """
        Construct the drift term for a process.
        """
        return Drift()
    
    def construct_difference_stationary(self, autoregression_parameters: np.ndarray) -> DifferenceStationary:
        """
        Construct the difference stationary term for a process.
        """
        return DifferenceStationary(n_steps=self.n_steps, autoregression_parameters=autoregression_parameters)
    
    def construct_diffusion(self, sigma: float, custom_noise: CustomNoise) -> Diffusion:
        """
        Construct the diffusion term for a process.
        """
        return Diffusion(n_steps=self.n_steps, sigma=sigma, custom_noise=custom_noise)
    
    def construct_jump(self, jump_type: JumpType, jump_params: Union[NormalPoissonJumpsParams, AssymetricBilateralJumpsParams, None],
                       custom_jump: CustomJump) -> Jump:
        """
        Construct the jump term for a process.
        """
        return Jump(n_steps=self.n_steps, jump_type=jump_type, jump_params=jump_params, custom_jump=custom_jump)

    def get_paths(self) -> np.ndarray:
        paths = []
        for _ in range(self.n_paths):
            paths.append(self.simulate_path())
        return np.array(paths)



def quick_build_stochastic_process() -> StochasticProcess:
    """
    Tick box style build of a customized stochastic process
    """
    # TODO: Complete quick_build_stochastic_process