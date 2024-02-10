from typing import (Union, Type)
import numpy as np
from src.digifi.utilities.general_utils import (type_check)
from src.digifi.stochastic_processes.sde_components_utils import (SDEComponentFunctionType, SDEComponentFunctionParams,
                                                              CustomSDEComponentFunction,
                                                              validate_custom_sde_component_function, get_component_values,
                                                              NoiseType, CustomNoise, JumpType, CompoundPoissonNormalParams,
                                                              CompoundPoissonBilateralParams, CustomJump)



class Drift:
    """
    Customizable drift term for a discrete-time SDE of the stochastic process.\n
    Creates a drift term of the form:
        Drift = component_func(previous_values, t) * dt
    """
    def __init__(self, n_paths: int, dt: float,
                 # Diffusion function parameters
                 component_type: SDEComponentFunctionType=SDEComponentFunctionType.LINEAR,
                 component_params: SDEComponentFunctionParams=SDEComponentFunctionParams(),
                 custom_drift: Union[Type[CustomSDEComponentFunction], None]=None) -> None:
        # Arguments validation
        type_check(value=component_params, type_=SDEComponentFunctionParams, value_name="component_params")
        type_check(value=component_type, type_=SDEComponentFunctionType, value_name="component_type")
        # Drift parameters
        self.n_paths = int(n_paths)
        self.dt = float(dt)
        self.component_type = component_type
        self.component_params = component_params
        if isinstance(custom_drift, CustomSDEComponentFunction):
            self.custom_drift = validate_custom_sde_component_function(custom_component=custom_drift, n_paths=self.n_paths, dt=self.dt)
        else:
            self.custom_drift = None
    
    def get_drift(self, previous_values: np.ndarray, t: float) -> np.ndarray:
        """
        Get an array of drift values for n paths.
        """
        # Get the values of the component in front of the dt
        component_values = get_component_values(component_type=self.component_type, component_params=self.component_params,
                                                stochastic_values=previous_values, t=t, n_paths=self.n_paths, dt=self.dt,
                                                custom_component_function=self.custom_drift)
        # Generate the drift term
        return component_values * self.dt



class Diffusion:
    """
    Customizable diffusion term for a discrete-time SDE of the stochastic process.\n
    Generates a diffusion term:
        Diffusion = component_func(previous_values, t) * noise_func
    """
    def __init__(self, n_paths: int, dt: float,
                 # Diffusion function parameters
                 component_type: SDEComponentFunctionType=SDEComponentFunctionType.LINEAR,
                 component_params: SDEComponentFunctionParams=SDEComponentFunctionParams(),
                 custom_diffusion: Union[Type[CustomSDEComponentFunction], None]=None,
                 # Noise parameters
                 noise_type: NoiseType=NoiseType.WEINER_PROCESS, custom_noise: Union[Type[CustomNoise], None]=None) -> None:
        # Arguments validation
        type_check(value=component_params, type_=SDEComponentFunctionParams, value_name="component_params")
        type_check(value=component_type, type_=SDEComponentFunctionType, value_name="component_type")
        type_check(value=noise_type, type_=NoiseType, value_name="noise_type")
        # Diffusion class parameters
        self.n_paths = int(n_paths)
        self.dt = float(dt)
        # Diffusion parameters
        self.component_type = component_type
        self.component_params = component_params
        if isinstance(custom_diffusion, CustomSDEComponentFunction):
            self.custom_diffusion = validate_custom_sde_component_function(custom_component=custom_diffusion, n_paths=self.n_paths, dt=self.dt)
        else:
            self.custom_diffusion = None
        # Noise parameters
        self.noise_type = noise_type
        if isinstance(custom_noise, CustomNoise):
            self.custom_noise = self.__validate_custom_noise(custom_noise=custom_noise)
        else:
            self.custom_noise = None
    
    def __validate_custom_noise(self, custom_noise: Type[CustomNoise]) -> Type[CustomNoise]:
        """
        Validate custom noise object to satisfy the computational requirements.
        """
        # Check that custom_noise takes in parameters dt and n_paths
        random_dt = float(np.random.uniform(low=0, high=1, size=1)[0])
        result = custom_noise.noise_func(n_paths=self.n_paths, dt=random_dt)
        # Check that custom_noise produces an array of length n_paths
        if len(result)!=self.n_paths:
            raise ValueError("The custom noise does not produce an array of length {}.".format(self.n_paths))
        return custom_noise
    
    def standard_white_noise(self) -> np.ndarray:
        """
        \\mathcal{N}(0, 1)\n
        Generate Gaussian white noise with variance 1.\n
        """
        return np.random.randn(self.n_paths)
    
    def weiner_process(self) -> np.ndarray:
        """
        dW_{t} \\approx \\sqrt{dt}*\\mathcal{N}(0, 1)\n
        Generate the Weiner process.\n
        Wikipedia: https://en.wikipedia.org/wiki/Wiener_process\n
        """
        return np.sqrt(self.dt) * self.standard_white_noise()
    
    def get_diffusion(self, previous_values: np.ndarray, t: float) -> np.ndarray:
        """
        Get an array of simulated diffusion values.
        """
        # Get the values of the component in front of the noise
        component_values = get_component_values(component_type=self.component_type, component_params=self.component_params,
                                                stochastic_values=previous_values, t=t, n_paths=self.n_paths, dt=self.dt,
                                                custom_component_function=self.custom_diffusion)
        # Generate the diffusion term
        match self.noise_type:
            case NoiseType.STANDARD_WHITE_NOISE:
                return component_values * self.standard_white_noise()
            case NoiseType.WEINER_PROCESS:
                return component_values * self.weiner_process()
            case NoiseType.CUSTOM_NOISE:
                return component_values * self.custom_noise.noise_func(n_paths=self.n_paths, dt=self.dt)



class Jump:
    """
    Customizable jump term for a discrete-time SDE of the stochastic process.
    """
    def __init__(self, n_paths: int, dt: float, jump_type: JumpType=JumpType.NO_JUMPS,
                 jump_params: Union[CompoundPoissonNormalParams, CompoundPoissonBilateralParams, None]=None,
                 custom_jump: Union[Type[CustomJump], None]=None) -> None:
        type_check(value=jump_type, type_=JumpType, value_name="jump_type")
        self.n_paths = int(n_paths)
        self.dt = float(dt)
        self.jump_type = jump_type
        self.jump_params = jump_params
        if isinstance(custom_jump, CustomJump):
            self.custom_jump = self.__validate_custom_jump(custom_jump=custom_jump)
        else:
            self.custom_jump = None
    
    def __validate_custom_jump(self, custom_jump: Type[CustomJump]) -> Type[CustomJump]:
        """
        Validate custom jump object to satisfy the computational requirements.
        """
        # Check that custom_jump takes in parameters dt and n_paths
        random_dt = np.random.uniform(low=0, high=1, size=1)[0]
        result = custom_jump.jump_func(n_paths=self.n_paths, dt=random_dt)
        # Check that custom_jump produces an array of length n_paths
        if len(result)!=self.n_paths:
            raise ValueError("The custom noise does not produce an array of length {}.".format(self.n_paths))
        return custom_jump
    
    def compound_poisson_normal(self, params: CompoundPoissonNormalParams) -> np.ndarray:
        """
        \\text{Jumps Frequency} = Pois(\\lambda dt)\n
        \\{Jumps Distribution} = (\\mu_{j} * \\text{Jumps Frequency}) + (\\sigma_{j} * \\text{Jumps Frequency} * \mathcal{N}(0,1))\n
        """
        # Frequency of jumps
        dP = np.random.poisson(params.lambda_j*float(self.dt), self.n_paths)
        # Distribution of jumps
        dJ = params.mu_j*dP + params.sigma_j*np.sqrt(dP)*np.random.randn(self.n_paths)
        return dJ
    
    def compound_poisson_bilateral(self, params: CompoundPoissonBilateralParams) -> np.ndarray:
        """
        \\text{Jumps Frequency} = Pois(\\lambda dt)\n
        \\text{Assymetric Double Exponential RV} = \mathbb{1}_{p\\leq U(0,1)}*(-\\frac{1}{\\eta_{u}} * ln(\\frac{1-U(0,1)}{p})) +\n
            + \mathbb{1}_{U(0,1)<p}*(\\frac{1}{\\eta_{d}} * ln(\\frac{U(0,1)}{1-p}))\n
        \\text{Jumps Distribution} = (e^{\\text{Assymetric Double Exponential RV}} - 1) * \\text{Jumps Frequency}\n
        """
        # Magnitude of jumps
        dP = np.random.poisson(params.lambda_j*float(self.dt), self.n_paths)
        # Assymetric double exponential random variable
        u = np.random.uniform(0, 1, self.n_paths)
        y = np.zeros(self.n_paths)
        for i in range(len(u)):
            if params.p<=u[i]:
                y[i] = (-1/params.eta_u) * np.log((1-u[i])/params.p)
            else:
                y[i] = (1/params.eta_d) * np.log(u[i]/(1-params.p))
        # Distribution of jumps
        dJ = (np.exp(y)-1)*dP
        return dJ

    def get_jumps(self) -> np.ndarray:
        """
        Get an array of simulated jump values.
        """
        match self.jump_type:
            case JumpType.NO_JUMPS:
                return np.zeros(self.n_paths)
            case JumpType.COMPOUND_POISSON_NORMAL:
                type_check(value=self.jump_params, type_=CompoundPoissonNormalParams, value_name="jump_params")
                return self.compound_poisson_normal(params=self.jump_params)
            case JumpType.COMPOUND_POISSON_BILATERAL:
                type_check(value=self.jump_params, type_=CompoundPoissonBilateralParams, value_name="jump_params")
                return self.compound_poisson_bilateral(params=self.jump_params)
            case JumpType.CUSTOM_JUMPS:
                return self.custom_jump.jump_func(n_paths=self.n_paths, dt=self.dt)