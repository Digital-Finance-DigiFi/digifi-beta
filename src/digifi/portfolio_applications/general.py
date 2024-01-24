from typing import Union
import abc
import enum
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.general_utils import verify_array



class ReturnsMethod(enum.Enum):
    IMPLIED_AVERAGE_RETURN = 1
    ESTIMATED_FROM_TOTAL_RETURN = 2



class ArrayRetrunsType(enum.Enum):
    RETURNS_OF_ASSETS = 1
    WEIGHTED_RETURNS_OF_ASSETS = 2
    PORTFOLIO_RETURNS = 3
    CUMULATIVE_PORTFOLIO_RETURNS = 4



@dataclass
class PortfolioInstrumentStruct:
    portfolio_price_array: np.ndarray
    portfolio_time_array: np.ndarray



def prices_to_returns(price_array: np.ndarray) -> np.ndarray:
    """
    Convert an array of prices to an array of returns.
    """
    verify_array(array=price_array, array_name="price_array")
    return np.diff(a=price_array)/price_array[1:]



def returns_average(price_array: np.ndarray, method: ReturnsMethod, n_periods: int=252) -> float:
    """
    Calculate the average return of a price array.
    """
    n_periods = int(n_periods)
    returns = prices_to_returns(price_array=price_array)
    match method:
        case ReturnsMethod.IMPLIED_AVERAGE_RETURN:
            mean_return = np.mean(returns)
            return ((1+mean_return)**n_periods)-1
        case ReturnsMethod.ESTIMATED_FROM_TOTAL_RETURN:
            compounded_return = (1 + returns).prod()
            return compounded_return**(n_periods/len(returns)) - 1
        case _:
            raise ValueError("The argument method must be of ReturnsMethod type.")



def returns_std(price_array: np.ndarray, n_periods: int=252) -> float:
    """
    Calculate the standard deviation of the returns of a price array.
    """
    n_periods = int(n_periods)
    returns_std = np.std(a=prices_to_returns(price_array=price_array))
    return returns_std * np.sqrt(n_periods)



def returns_variance(price_array: np.ndarray, n_periods: int=252) -> float:
    """
    Calculate the variance of the returns of a price array.
    """
    n_periods = int(n_periods)
    returns_variance = np.std(a=prices_to_returns(price_array=price_array))**2
    return returns_variance * n_periods



class PortfolioInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "change_weights") and
                callable(subclass.change_weights) and
                hasattr(subclass, "add_asset") and
                callable(subclass.add_asset) and
                hasattr(subclass, "remove_asset") and
                callable(subclass.remove_asset) and
                hasattr(subclass, "array_returns") and
                callable(subclass.array_returns) and
                hasattr(subclass, "mean_return") and
                callable(subclass.mean_return) and
                hasattr(subclass, "covariance") and
                callable(subclass.covariance) and
                hasattr(subclass, "standard_deviation") and
                callable(subclass.standard_deviation) and
                hasattr(subclass, "autocorrelation") and
                callable(subclass.autocorrelation) and
                hasattr(subclass, "sharpe_ratio") and
                callable(subclass.sharpe_ratio) and
                hasattr(subclass, "maximize_sharpe_ratio") and
                callable(subclass.maximize_sharpe_ratio) and
                hasattr(subclass, "minimize_std") and
                callable(subclass.minimize_std) and
                hasattr(subclass, "efficient_optimization") and
                callable(subclass.efficient_optimization))
    
    @abc.abstractmethod
    def change_weights(self, new_weights: np.ndarray) -> None:
        """
        Update weights of the portfolio assets.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def add_asset(self, identifier: str) -> None:
        """
        Add asset to portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def remove_asset(self, identifier: str) -> None:
        """
        Remove asset from portfolio.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def array_returns(self, operation_type: ArrayRetrunsType) -> Union[dict[np.ndarray], np.ndarray]:
        """
        Calculate returns for the provided operation type.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def mean_return(self) -> float:
        """
        Calculate the mean return of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Calculate the covariance of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def standard_deviation(self) -> float:
        """
        Calculate the standard deviation of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def autocorrelation(self) -> np.ndarray:
        """
        Calculate the autocorrelation of portfolio returns.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def sharpe_ratio(self) -> float:
        """
        Sharpe ratio = (portfolio returns - risk-free rate) / portfolio standard deviation
        Calculate the Sharpe ratio of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def maximize_sharpe_ratio(self) -> float:
        """
        Find portfolio with maximum Sharpe ratio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def minimize_std(self) -> float:
        """
        Find portfolio with lowest standard deviation.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def efficient_optimization(self) -> float:
        """
        Find risk level on the efficient frontier for a given target return.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def efficient_frontier(self) -> dict:
        """
        Calculate efficient frontier.
        """
        raise NotImplementedError