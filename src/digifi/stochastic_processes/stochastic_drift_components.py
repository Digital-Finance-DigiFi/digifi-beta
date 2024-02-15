from typing import (Union, Type)
import abc
from enum import Enum
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.general_utils import (type_check, DataClassValidation)



class StochasticDriftType(Enum):
    TREND_STATIONARY = 1
    DIFFERENCE_STATIONARY = 2



class TrendStationaryTrendType(Enum):
    LINEAR = 1
    QUADRATIC = 2
    EXPONENTIAL = 3
    CUSTOM = 4



@dataclass(slots=True)
class TrendStationaryParams(DataClassValidation):
    linear_a: float = 1.0
    linear_b: float = 1.0
    quadratic_a: float = 1.0
    quadratic_b: float = 1.0
    quadratic_c: float = 1.0
    exponential_a: float = 1.0
    exponential_b: float = 1.0

    def validate_quadratic_a(sel, value: float, **_) -> float:
        if value==0:
            raise ValueError("The argument quadratic_a must not be 0.")
        return value



class CustomTrendStationaryFunction(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "trend_func") and
                callable(subclass.trend_func))
    
    @abc.abstractmethod
    def trend_func(self, t: float, n_paths: int) -> np.ndarray:
        """
        Custom trend-stationary component function.
        """
        raise NotImplementedError



class StationaryErrorType(Enum):
    WEINER = 1
    CUSTOM = 2



class CustomError(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "error_func") and
                callable(subclass.error_func))
    
    @abc.abstractmethod
    def error_func(self, n_paths: int, dt: float) -> np.ndarray:
        """
        Custom error component function.
        """
        raise NotImplementedError



class DifferenceStationary:
    """
    Difference stationary term for a discrte-time stochastic process.\n
    Wikipedia: https://en.wikipedia.org/wiki/Unit_root \n
    """
    def __init__(self, n_paths: int, autoregression_parameters: np.ndarray) -> None:
        type_check(value=autoregression_parameters, type_=np.ndarray, value_name="autoregression_parameters")
        self.n_paths = int(n_paths)
        self.autoregression_parameters = autoregression_parameters
    
    def get_autoregression(self, previous_values: list[np.ndarray], t: float) -> np.ndarray:
        """
        Get an array of autoregression values.
        """
        process_values = np.array(previous_values).T
        # Arguments validation
        if len(process_values)!=self.n_paths:
            raise ValueError("The variable process_values does not produce an array of length {} defined by n_paths.".format(self.n_paths))
        # Perform autoregression for the next term in each path
        process_order = len(self.autoregression_parameters)
        result = np.array([])
        for i in range(self.n_paths):
            product = np.dot(self.autoregression_parameters, process_values[i][-process_order:])
            result = np.append(result, product)
        return result



class TrendStationary:
    """
    Trend stationary term for a discrete-time stochastic process.\n
    Wikipedia: https://en.wikipedia.org/wiki/Trend-stationary_process\n
    """
    def __init__(self, n_paths: int, trend_type: TrendStationaryTrendType=TrendStationaryTrendType.LINEAR,
                 trend_params: TrendStationaryParams=TrendStationaryParams(),
                 custom_trend: Union[CustomTrendStationaryFunction, None]=None) -> None:
        # Arguments validation
        type_check(value=trend_type, type_=TrendStationaryTrendType, value_name="trend_type")
        type_check(value=trend_params, type_=TrendStationaryParams, value_name="trend_params")
        self.n_paths = int(n_paths)
        self.trend_type = trend_type
        self.trend_params = trend_params
        if isinstance(custom_trend, CustomTrendStationaryFunction):
            self.custom_trend = self.__validate_custom_trend(custom_trend=custom_trend)
        else:
            self.custom_trend = None
    
    def __validate_custom_trend(self, custom_trend: Type[CustomTrendStationaryFunction]):
        """
        Validate custom trend-stationary component function object to satisfy the computational requirements.
        """
        # Check that trend_func takes in parameters t and n_paths
        t = float(np.random.uniform(low=0, high=1, size=1)[0])
        result = custom_trend.trend_func(t=t, n_paths=self.n_paths)
        # Check that comp_func produces an array of length n_paths
        if len(result)!=self.n_paths:
            raise ValueError("The custom trend function does not produce an array of length {} defined by n_paths.".format(self.n_paths))
        return custom_trend

    def get_stationary_trend(self, t: float) -> np.ndarray:
        """
        Get an array of trend-stationary trend values.\n
        Returns a trend value for a given stationary trend type:
            - Linear Trend (params: a, b): a*t + b + error_term
            - Quadratic Trend (params: a, b): a*t**2 + b*t + c + error_term
            - Exponential Trend (params: a, b): b * exp(a*t) * error_term
        """
        base_shape = np.ones(self.n_paths)
        t = float(t)
        # Resolve trend type
        match self.trend_type:
            case TrendStationaryTrendType.LINEAR:
                return (self.trend_params.linear_a*t + self.trend_params.linear_b) * base_shape
            case TrendStationaryTrendType.QUADRATIC:
                return (self.trend_params.quadratic_a*(t**2) + self.trend_params.quadratic_b*t + self.trend_params.quadratic_c) * base_shape
            case TrendStationaryTrendType.EXPONENTIAL:
                return (self.trend_params.exponential_b*np.exp(self.trend_params.exponential_a*t)) * base_shape
            case TrendStationaryTrendType.CUSTOM:
                return self.custom_trend.trend_func(t=t, n_paths=self.n_paths)



class StationaryError:
    """
    Stationary error term for a discrete-time stochastic process.
    """
    def __init__(self, n_paths: int, dt: float, error_type: StationaryErrorType=StationaryErrorType.WEINER, sigma: float=1.0,
                 custom_error: Union[Type[CustomError], None]=None) -> None:
        type_check(value=error_type, type_=StationaryErrorType, value_name="error_type")
        self.n_paths = int(n_paths)
        self.dt = float(dt)
        self.error_type = error_type
        self.sigma = float(sigma)
        if isinstance(custom_error, CustomError):
            self.custom_error = self.__validate_custom_error(custom_error=custom_error)
        else:
            self.custom_error = None
    
    def __validate_custom_error(self, custom_error: Type[CustomError]) -> Type[CustomError]:
        """
        Validate custom error component function object to satisfy the computational requirements.
        """
        # Check that error_func takes in parameters t and n_paths
        result = custom_error.error_func(n_paths=self.n_paths, dt=self.dt)
        # Check that error_func produces an array of length n_paths
        if len(result)!=self.n_paths:
            raise ValueError("The custom error function does not produce an array of length {} defined by n_paths.".format(self.n_paths))
        return custom_error
    
    def weiner_process(self) -> np.ndarray:
        """
        Generate increments of the Weiner process.\n
        Wikipedia: https://en.wikipedia.org/wiki/Wiener_process\n
        """
        return (self.sigma**2) * self.dt * np.random.randn(self.n_paths)
    
    def get_error(self) -> np.ndarray:
        """
        Get an array of simulated error values.
        """
        match self.error_type:
            case StationaryErrorType.WEINER:
                return self.weiner_process()
            case StationaryErrorType.CUSTOM:
                return self.custom_error.error_func(n_paths=self.n_paths, dt=self.dt)