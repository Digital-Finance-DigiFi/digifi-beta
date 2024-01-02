from typing import Union, List
import copy
import numpy as np
import scipy.optimize as sc
from src.digifi.portfolio_applications.general import (ReturnsMethod, ArrayRetrunsType, prices_to_returns, returns_average,
                                                       PortfolioInterface)
from src.digifi.utilities.general_utils import (verify_array, compare_array_len)
# TODO: Add portfolio autocorrelation



class Portfolio(PortfolioInterface):
    """
    Portfolio of assets.
    """
    def __init__(self, assets: dict[np.ndarray], weights: np.ndarray) -> None:
        # Assets type validation
        test_asset_id, test_asset = self.__select_test_asset(assets=assets)
        for key in list(assets.keys()):
            compare_array_len(array_1=test_asset, array_2=assets[key], array_1_name=test_asset_id, array_2_name=key)
        self.assets = assets
        # Weights type validation
        self.change_weights(weights=weights)
    
    def __select_test_asset(self, assets: dict[np.ndarray]) -> (str, np.ndarray):
        return (list(assets.keys())[0], list(assets.values())[0])
    
    def __tensorize_assets(self) -> dict[list[str], list[np.ndarray]]:
        """
        Convert assets from a dictionary to a matrix.
        """
        assets_ids = list()
        assets_prices = list()
        assets_keys = list(self.assets.keys())
        assets_values = list(self.assets.values())
        for i in range(len(self.assets)):
            assets_ids.append(assets_keys[i])
            assets_prices.append(copy.deepcopy(assets_values[i]))
        return {"assets_ids":assets_ids, "assets_prices":assets_prices}
    
    def __untensorize_assets(self, assets_ids: List[str], assets_arrays: np.ndarray) -> dict[np.ndarray]:
        if len(assets_ids)!=len(assets_arrays):
            raise ValueError("The length of assets_ids and assets_prices do not match.")
        assets_dict = dict()
        for i in range(len(assets_ids)):
            assets_dict[assets_ids[i]] = assets_arrays[i]
        return assets_dict
    
    def change_weights(self, weights: np.ndarray) -> None:
        """
        Update weights of the portfolio assets.
        """
        verify_array(array=weights, array_name="weights")
        if sum(weights)!=1:
            raise ValueError("The weights must add up to 1.")
        if len(weights)!=len(self.assets):
            raise ValueError("The number of weights has to coincide with the number of assets provided.")
        self.weights = weights
    
    def add_asset(self, new_asset_identifier: str, new_asset_prices: np.ndarray) -> None:
        """
        Add asset to portfolio.
        """
        new_asset_identifier = str(new_asset_identifier)
        test_asset_id, test_asset = self.__select_test_asset(assets=self.assets)
        compare_array_len(array_1=test_asset, array_2=new_asset_prices, array_1_name=test_asset_id, array_2_name=new_asset_identifier)
        if new_asset_identifier in list(self.assets.keys()):
            raise ValueError("An asset with the identifier {} already exists in the portfolio.".format(new_asset_identifier))
        self.assets[new_asset_identifier] = new_asset_prices
    
    def remove_asset(self, asset_identifier: str) -> None:
        """
        Remove asset from portfolio.
        """
        if asset_identifier in list(self.assets.keys()):
            del self.assets[asset_identifier]
   
    def array_returns(self, operation_type: ArrayRetrunsType=ArrayRetrunsType.CUMULATIVE_PORTFOLIO_RETURNS,
                      untensorize_result: bool=True) -> Union[dict[np.ndarray], tuple[list, np.ndarray], np.ndarray]:
        """
        Calculate returns for the provided operation type.
        """
        tensorized_result = self.__tensorize_assets()
        assets_ids, assets_arrays = tensorized_result["assets_ids"], tensorized_result["assets_prices"]
        for i in range(len(assets_arrays)):
            assets_arrays[i] = prices_to_returns(price_array=assets_arrays[i])
        # Returns of assets
        if operation_type==ArrayRetrunsType.RETURNS_OF_ASSETS:
            if untensorize_result:
                return self.__untensorize_assets(assets_ids=assets_ids, assets_arrays=assets_arrays)
            else:
                return (assets_ids, assets_arrays)
        # Weighted returns of assets
        for i in range(len(assets_arrays)):
            assets_arrays[i] = assets_arrays[i] * self.weights[i]
        if operation_type==ArrayRetrunsType.WEIGHTED_RETURNS_OF_ASSETS:
            if untensorize_result:
                return self.__untensorize_assets(assets_ids=assets_ids, assets_arrays=assets_arrays)
            else:
                return (assets_ids, assets_arrays)
        # Portfolio returns
        assets_arrays = np.sum(assets_arrays, axis=0)
        if operation_type==ArrayRetrunsType.PORTFOLIO_RETURNS:
            return assets_arrays
        assets_arrays = (np.cumprod(1 + assets_arrays) - 1)
        return assets_arrays
    
    def mean_return(self, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN) -> float:
        """
        Calculate the mean return of the portfolio.
        """
        n_periods = int(n_periods)
        tensorized_result = self.__tensorize_assets()
        assets_ids, assets_arrays = tensorized_result["assets_ids"], tensorized_result["assets_prices"]
        for i in range(len(assets_arrays)):
            assets_arrays[i] = returns_average(price_array=assets_arrays[i], method=method, n_periods=n_periods) * self.weights[i]
        return np.sum(assets_arrays)
    
    def covariance(self, n_periods: int=252, untensorize_result: bool=True) -> np.ndarray:
        """
        Calculate the covariance of the portfolio.
        """
        n_periods = int(n_periods)
        assets_ids, assets_returns = self.array_returns(operation_type=ArrayRetrunsType.RETURNS_OF_ASSETS, untensorize_result=False)
        covariance_matrix = np.cov(assets_returns, ddof=0)*n_periods
        if untensorize_result:
            return self.__untensorize_assets(assets_ids=assets_ids, assets_arrays=covariance_matrix)
        return covariance_matrix
    
    def standard_deviation(self, n_periods: int=252) -> float:
        """
        Calculate the standard deviation of the portfolio.
        """
        n_periods = int(n_periods)
        covariance_matrix = self.covariance(n_periods=n_periods, untensorize_result=False)
        return np.sqrt(np.dot(self.weights.T, np.dot(covariance_matrix, self.weights)))
    
    def sharpe_ratio(self, r: float, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN) -> float:
        """
        Sharpe ratio = (portfolio returns - risk-free rate) / portfolio standard deviation
        Calculate the Sharpe ratio of the portfolio.
        """
        r = float(r)
        n_periods = int(n_periods)
        portfolio_return = self.mean_return(n_periods=n_periods, method=method)
        portfolio_std = self.standard_deviation(n_periods=n_periods)
        return (portfolio_return - r) / portfolio_std

    def __unsafe_change_weights(self, weights: np.ndarray) -> None:
        """
        Unsafe change of weights for the purpose of numerical solution only. Does not check whether weights sum up to 1.
        """
        verify_array(array=weights, array_name="weights")
        if len(weights)!=len(self.assets):
            raise ValueError("The number of weights has to coincide with the number of assets provided.")
        self.weights = weights

    def __proxy_sharpe_ratio(self, weights: np.ndarray, r: float, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN) -> float:
        """
        Helper method to define Sharpe ratio optimization in terms of weights.
        """
        self.__unsafe_change_weights(weights=weights)
        return -1*self.sharpe_ratio(r=r, n_periods=n_periods, method=method)
    
    def __proxy_mean_return(self, weights: np.ndarray, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN) -> float:
        """
        Helper method to define mean return in terms of weights.
        """
        self.__unsafe_change_weights(weights=weights)
        return self.mean_return(n_periods=n_periods, method=method)
    
    def __proxy_std(self, weights: np.ndarray, n_periods: int=252) -> float:
        """
        Helper method to define standard deviation optimization in terms of weights.
        """
        self.__unsafe_change_weights(weights=weights)
        return self.standard_deviation(n_periods=n_periods)
    
    def maximize_sharpe_ratio(self, r: float, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN,
                              weight_constraint: tuple=(0,1), verbose: bool=False) -> float:
        """
        Find portfolio with maximum Sharpe ratio.
        """
        r = float(r)
        n_periods = int(n_periods)
        n_assets = len(self.assets)
        args = (r, n_periods, method)
        bounds = tuple(weight_constraint for _ in range(n_assets))
        constraints = ({"type":"eq", "fun": lambda x: np.sum(x) - 1})
        result = sc.minimize(fun=self.__proxy_sharpe_ratio, x0=np.ones(n_assets)/n_assets, args=args, method="SLSQP", bounds=bounds,
                             constraints=constraints)
        self.__unsafe_change_weights(weights=result["x"])
        if verbose:
            print("Portfolio weights: {}".format(result["x"]))
            print("Maximum Sharpe ratio: {}".format(-1*result["fun"]))
        return -1*result["fun"]
    
    def minimize_std(self, n_periods: int=252, weight_constraint: tuple=(0,1), verbose: bool=False) -> float:
        """
        Find portfolio with lowest standard deviation.
        """
        n_periods = int(n_periods)
        n_assets = len(self.assets)
        args = (n_periods)
        bounds = tuple(weight_constraint for _ in range(n_assets))
        constraints = ({"type":"eq", "fun": lambda x: np.sum(x) - 1})
        result = sc.minimize(fun=self.__proxy_std, x0=np.ones(n_assets)/n_assets, args=args, method="SLSQP", bounds=bounds,
                             constraints=constraints)
        self.__unsafe_change_weights(weights=result["x"])
        if verbose:
            print("Portfolio weights: {}".format(result["x"]))
            print("Minimum standard deviation: {}".format(result["fun"]))
        return result["fun"]
    
    def efficient_optimization(self, return_target: float, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN,
                               weight_constraint: tuple=(0,1), verbose: bool=False) -> float:
        """
        Find risk level on the efficient frontier for a given target return.
        """
        return_target = float(return_target)
        n_periods = int(n_periods)
        n_assets = len(self.assets)
        args = (n_periods)
        bounds = tuple(weight_constraint for _ in range(n_assets))
        constraints = ({"type":"eq", "fun": lambda x: self.__proxy_mean_return(weights=x, n_periods=n_periods, method=method) - return_target},
                       {"type":"eq", "fun": lambda x: np.sum(x) - 1})
        result = sc.minimize(fun=self.__proxy_std, x0=np.ones(n_assets)/n_assets, args=args, method="SLSQP", bounds=bounds,
                             constraints=constraints)
        self.__unsafe_change_weights(weights=result["x"])
        if verbose:
            print("Portfolio weights: {}".format(result["x"]))
            print("Efficient risk level: {}".format(result["fun"]))
        return result["fun"]
    
    def efficient_frontier(self, r: float, frontier_n_points: int=20, n_periods: int=252, method: ReturnsMethod=ReturnsMethod.IMPLIED_AVERAGE_RETURN,
                           weight_constraint: tuple=(0,1)) -> dict:
        """
        Calculate efficient frontier.
        """
        frontier_n_points = int(frontier_n_points)
        r = float(r)
        n_periods = int(n_periods)
        # Minimum volatility portfolio
        min_vol_std = self.minimize_std(n_periods=n_periods, weight_constraint=weight_constraint)
        min_vol_mean_return = self.mean_return(n_periods=n_periods, method=method)
        # Maximum Sharpe ratio portfolio
        self.maximize_sharpe_ratio(r=r, n_periods=n_periods, method=method, weight_constraint=weight_constraint)
        max_sr_mean_return = self.mean_return(n_periods=n_periods, method=method)
        max_sr_std = self.standard_deviation(n_periods=n_periods)
        # Efficient frontier between min-vol portfolio and max Sharpe ratio portfolio
        efficient_risk = np.array([])
        target_returns = np.linspace(start=min_vol_mean_return, stop=max_sr_mean_return, num=frontier_n_points)
        for target in target_returns:
            efficient_risk = np.append(efficient_risk, self.efficient_optimization(return_target=target, n_periods=n_periods, method=method,
                                                                                   weight_constraint=weight_constraint))
        return {"min_vol":{"return":min_vol_mean_return, "std":min_vol_std},
                "eff":{"return":target_returns, "std":efficient_risk},
                "max_sr":{"return":max_sr_mean_return, "std":max_sr_std}}



class InstrumentsPortfolio(PortfolioInterface):
    """
    Portfolio that consists of financial instuments that have PortfolioInstumentStruct defined for them.
    """
    def __init__(self, instruments: List) -> None:
        # TODO: Implement InstumentPortfolio and PortfolioInstrumentStruct for all financial instruments
        pass