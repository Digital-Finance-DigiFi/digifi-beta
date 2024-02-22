from typing import (Union, Type)
import abc
from enum import Enum
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from digifi.utilities.general_utils import (compare_array_len, type_check, DataClassValidation)
from digifi.probability_distributions.general import ProbabilityDistributionInterface



class ReturnsMethod(Enum):
    IMPLIED_AVERAGE_RETURN = 1
    ESTIMATED_FROM_TOTAL_RETURN = 2



class ArrayRetrunsType(Enum):
    RETURNS_OF_ASSETS = 1
    WEIGHTED_RETURNS_OF_ASSETS = 2
    PORTFOLIO_RETURNS = 3
    CUMULATIVE_PORTFOLIO_RETURNS = 4



class PortfolioOptimizationResultType(Enum):
    VALUE = 1
    WEIGHTS = 2



class PortfolioInstrumentResultType(Enum):
    ARRAY = 1
    FLOAT = 2



@dataclass(slots=True)
class PortfolioInstrumentStruct(DataClassValidation):
    """
    ## Description
    Struct with data to be used inside the InstumentsPortfolio.
    ### Input:
        - portfolio_price_array (np.ndarray): Historical price series of the instrument
        - portfolio_time_array (np.ndarray): An array of time accompanying the price series
        - portfolio_predictable_income (np.ndarray): An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    """
    portfolio_price_array: np.ndarray
    portfolio_time_array: np.ndarray
    portfolio_predicatble_income: np.ndarray

    def __post_init__(self) -> None:
        super(PortfolioInstrumentStruct, self).__post_init__()
        compare_array_len(array_1=self.portfolio_price_array, array_2=self.portfolio_time_array,
                          array_1_name="portfolio_prices_array", array_2_name="portfolio_time_array")
        compare_array_len(array_1=self.portfolio_price_array, array_2=self.portfolio_predicatble_income,
                          array_1_name="portfolio_price_array", array_2_name="portfolio_predictable_income")

    def get_price(self, t: int, data_format: PortfolioInstrumentResultType=PortfolioInstrumentResultType.ARRAY) -> Union[np.ndarray, float]:
        """
        ## Description
        Safe method for working with historical price data.\n
        This method prevents the user from using the future prices based on the time value provided.
        ### Input:
            - t (int): Time index beyond which no data will be returned
            - data_format (PortfolioInstrumentResultType): Format of the returned datd (i.e., return an array of the historical data of a float value)
        ### Output:
            - Historical and/or current prices(s) (Union[np.ndarray, float])
        """
        # Arguments validation
        type_check(value=data_format, type_=PortfolioInstrumentResultType, value_name="data_format")
        t = int(t)
        if t<len(self.portfolio_time_array):
            match data_format:
                case PortfolioInstrumentResultType.ARRAY:
                    return self.portfolio_price_array[:t]
                case PortfolioInstrumentResultType.FLOAT:
                    return self.portfolio_price_array[t]
        else:
            raise ValueError("The argument t must be an index in the range [0, {}]".format(len(self.portfolio_time_array)))

    def get_predictable_income(self, t: int, data_format: PortfolioInstrumentResultType=PortfolioInstrumentResultType.ARRAY) -> Union[np.ndarray, float]:
        """
        ## Description
        Safe method for working with historical predictable income data.\n
        This method prevents the user from using the future predictable incomes based on the time value provided.
        ### Input:
            - t (int): Time index beyond which no data will be returned
            - data_format (PortfolioInstrumentResultType): Format of the returned datd (i.e., return an array of the historical data of a float value)
        ### Output:
            - Historical and/or current predictable income(s) (Union[np.ndarray, float])
        """
        # Arguments validation
        type_check(value=data_format, type_=PortfolioInstrumentResultType, value_name="data_format")
        t = int(t)
        if t<len(self.portfolio_time_array):
            match data_format:
                case PortfolioInstrumentResultType.ARRAY:
                    return self.portfolio_predicatble_income[:t]
                case PortfolioInstrumentResultType.FLOAT:
                    return self.portfolio_predicatble_income[t]
        else:
            raise ValueError("The argument t must be an index in the range [0, {}]".format(len(self.portfolio_time_array)))



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
        ## Description
        Update weights of the portfolio assets.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def add_asset(self, identifier: str) -> None:
        """
        ## Description
        Add asset to portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def remove_asset(self, identifier: str) -> None:
        """
        ## Description
        Remove asset from portfolio.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def array_returns(self, operation_type: ArrayRetrunsType) -> Union[dict[np.ndarray], np.ndarray]:
        """
        ## Description
        Calculate returns for the provided operation type.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def mean_return(self) -> float:
        """
        ## Description
        Calculate the mean return of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def covariance(self) -> np.ndarray:
        """
        ## Description
        Calculate the covariance of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def standard_deviation(self) -> float:
        """
        ## Description
        Calculate the standard deviation of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def autocorrelation(self) -> np.ndarray:
        """
        ## Description
        Calculate the autocorrelation of portfolio returns.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def sharpe_ratio(self) -> float:
        """
        ## Description
        Sharpe ratio = (portfolio returns - risk-free rate) / portfolio standard deviation
        Calculate the Sharpe ratio of the portfolio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def maximize_sharpe_ratio(self) -> float:
        """
        ## Description
        Find portfolio with maximum Sharpe ratio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def minimize_std(self) -> float:
        """
        ## Description
        Find portfolio with lowest standard deviation.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def efficient_optimization(self) -> float:
        """
        ## Description
        Find risk level on the efficient frontier for a given target return.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def efficient_frontier(self) -> dict:
        """
        ## Description
        Calculate efficient frontier.
        """
        raise NotImplementedError



def volatility_from_historical_data(price_array: np.ndarray, n_periods: int) -> float:
    """
    ## Description
    Calculate the volatility of a price array from historical data.\n
    Note: There must be fixed time intervals between prices, and the distribution of prices is considered to be log-normal.
    ### Input:
        - price_array (np.ndarray): Price time series
        - n_periods (int): Number of periods used to estimate the volatility over (e.g., for daily prices n_periods=252 produces annualized volatility)
    ### Output:
        - Volatility (float) of price over a certain period
    """
    # Arguments validation
    n_periods = int(n_periods)
    type_check(value=price_array, type_=np.ndarray, value_name="price_array")
    # Historical volatility calculation
    std = np.std(np.log(price_array[1:]) - np.log(price_array[:-1]))
    return std*np.sqrt(n_periods)



class RiskMeasures:
    """
    ## Description
    Static methods for calculating the risk of a portfolio.
    """
    @staticmethod
    def value_at_risk(alpha: float, returns_distribution: Type[ProbabilityDistributionInterface]) -> float:
        """
        ## Description
        Measure of the risk of a portfolio estimating how much a portfolio can lose in a specified period.\n
        Note: This function uses the convention where 5% V@R of $1 million is the 0.05 probability that the portfolio will go down by $1 million.\n
        Note: The V@R is quoted as a positive number, if V@R is negative it implies that the portfolio has very high chace of making a profit.
        ### Input:
            - alpha (float): Probability level for V@R
            - returns_distribution (Type[ProbabilityDistributionInterface]): Probability distribution object with an inverse CDF method
        ### Output:
            - Value at risk (V@R) (float)
        ### links:
            - Wikipedia: https://en.wikipedia.org/wiki/Value_at_risk#
            - Original Source: N/A
        """
        # Arguments validation
        alpha = float(alpha)
        if (alpha<0) or (1<alpha):
            raise ValueError("The argument alpha must be in the range[0, 1].")
        type_check(value=returns_distribution, type_=ProbabilityDistributionInterface, value_name="returns_distribution")
        # Value at risk
        return -returns_distribution.inverse_cdf(p=np.array([alpha]))[0]

    @staticmethod
    def expected_shortfall(alpha: float, returns_distribution: Type[ProbabilityDistributionInterface]) -> float:
        """
        ## Description
        Measure of the risk of a portfolio that evaluates the expected return of a portfolio in the worst percentage of cases.\n
        Note: This function uses the convention where ES at 5% is the expected shortfall of the 5% of worst cases.
        ### Input:
            - alpha (float): Probability level for ES
            - returns_distribution (Type[ProbabilityDistributionInterface]): Probability distribution object with an inverse CDF method
        ### Output:
            - Expected shortfall (ES) (float)
        ### links:
            - Wikipedia: https://en.wikipedia.org/wiki/Expected_shortfall
            - Original Source: N/A
        """
        # Arguments validation
        alpha = float(alpha)
        if (alpha<0) or (1<alpha):
            raise ValueError("The argument p_es must be in the range[0, 1].")
        type_check(value=returns_distribution, type_=ProbabilityDistributionInterface, value_name="returns_distribution")
        # Expected shortfall
        return quad(lambda p: RiskMeasures().value_at_risk(alpha=p, returns_distribution=returns_distribution), 0, alpha)[0]/alpha



class UtilityFunctions:
    """
    ## Description
    Static methods for calculating the utility of consumption/wealth.
    """
    @staticmethod
    def cara(consumption: float, absolute_risk_aversion: float) -> float:
        """
        ## Description
        Exponential utility is a constant absolute risk aversion (CARA) utility measure with respect to consumption.
        ### Input:
            - consumption (float): Wealth or goods being measured
            - absolute_risk_aversion (float): Parameter determining how risk averse the utility function is
        ### Output:
            - Utility (float)
        ### LaTeX Formula:
            - u(c) = 1 - e^{-\\alpha c}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Risk_aversion#Absolute_risk_aversion
            - Original Source: N/A
        """
        return 1 - np.exp(-float(absolute_risk_aversion)*float(consumption))

    @staticmethod
    def crra(consumption: float, relative_risk_aversion: float) -> float:
        """
        ## Description
        Isoelastic utility is a constant relative risk aversion (CRRA) utility measure with respect to consumption.
        ### Input:
            - consumption (float): Wealth or goods being measured
            - relative_risk_aversion (float): Parameter determining how risk averse the utility function is
        ### Output:
            - Utility (float)
        ### LaTeX Formula:
            - u(c) = \\frac{c^{1-\\rho} - 1}{1 - \\rho}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Risk_aversion#Relative_risk_aversion
            - Original Source: N/A
        """
        rho = float(relative_risk_aversion)
        return (float(consumption)**(1-rho) - 1) / (1 - rho)



class PortfolioPerformance:
    """
    ## Description
    Static methods for calculating portfolio/asset performance.
    """
    @staticmethod
    def sharpe_ratio(portfolio_return: float, rf: float, portfolio_std: float) -> float:
        """
        ## Description
        Measure of the performance of a portfolio compared to risk-free rate and adjusted for risk.
        ### Input:
            - portfolio_return (float): Expected return of the portfolio
            - rf (float): Risk-free rate of return
            - portfolio_std (float): Standard deviation of portfolio returns
        ### Output:
            - Sharpe raio (float)
        ### LaTeX Formula:
            - S_{P} = \\frac{E[R_{P}]-r_{f}}{\\sigma^{2}_{P}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Sharpe_ratio
            - Original Source: https://doi.org/10.1086%2F294846
        """
        return (float(portfolio_return) - float(rf)) / float(portfolio_std)

    @staticmethod
    def information_ratio(portfolio_sharpe: float, benchmark_sharpe: float) -> float:
        """
        ## Description
        Measure of the performance of a portfolio compared to a benchmark relative to the volatility of the active return.
        ### Input:
            - portfolio_sharpe (float): Sharpe ratio of the portfolio
            - benchmark_sharpe (float): Sharpe ratio of the benchmark portfolio
        ### Output
            - Information ratio (float)
        ### LaTeX Formula:
            - IR = \\frac{E[R_{P}-R_{B}]}{\\sqrt{Var[R_{P}-R_{B}]}} = \\sqrt{S^{2}_{P} - S^{2}_{B}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Information_ratio
            - Original Source: N/A
        """
        return np.sqrt(float(portfolio_sharpe)**2 - float(benchmark_sharpe)**2)

    @staticmethod
    def treynor_ratio(portfolio_return: float, rf: float, portfolio_beta: float) -> float:
        """
        ## Description
        Measure of the performance of a portfolio in excess of what could have been earned on an investment with no diversifiable risk.
        ### Input:
            - portfolio_return (float): Expected return of the portfolio
            - rf (float): Risk-free rate of return
            - portfolio_beta (float): Beta of the portfolio with respect to the market returns
        ### Output:
            - Treynor ratio (float)
        ### LaTeX Formula:
            - T_{P}  =\\frac{E[R_{P}]-r_{f}}{\\beta_{P}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Treynor_ratio
            - Original Source: N/A
        """
        return (float(portfolio_return - float(rf))) / float(portfolio_beta)

    @staticmethod
    def jensens_alpha(portfolio_return: float, rf: float, portfolio_beta: float, market_return: float) -> float:
        """
        ## Description
        Measure of the performance of the portfolio in excess of its theoretical expected return.
        ### Input:
            - portfolio_return (float): Actual portfolio return
            - rf (float): Risk-free rate of return
            - portfolio_beta (float): Beta of the portfolio with respect to the market returns
            - market_return (float): Expected return of the market
        ### Output:
            - Jensen's alpha (float)
        ### LaTeX Formula:
            - \\alpha_{J} = R_{P} - (r_{f} + \\beta_{P}(R_{M} - r_{f}))
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Jensen%27s_alpha
            - Original Source: https://dx.doi.org/10.2139/ssrn.244153
        """
        rf = float(rf)
        return float(portfolio_return) - rf - float(portfolio_beta)*(float(market_return)-rf)

    @staticmethod
    def sortino_ratio(portfolio_realized_return: float, target_return: float, downside_risk: float) -> float:
        """
        ## Description
        Measure of the performance of the portfolio in the form of risk-adjusted return.\n
        This measure is the extension of Sharpe ratio, but it penalizes the 'downside' volatility and not the 'upside' volatility.
        ### Input:
            - portfolio_realized_return (float): Realized return of the portfolio
            - target_return (float): Target or required return of the portfolio
            - downside_risk (float): Downside risk (Standard deviation of the negative returns only)
        ### Output:
            - Sortino ratio (float)
        ### LaTeX Formula:
            - S = \\frac{R_{P} - T}{DR}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Sortino_ratio
            - Original Source: https://doi.org/10.3905%2Fjoi.3.3.59
        """
        return (float(portfolio_realized_return) - float(target_return)) / float(downside_risk)



def prices_to_returns(price_array: np.ndarray) -> np.ndarray:
    """
    ## Description
    Convert an array of prices to an array of returns.
    ### Input:
        - price_array (np.ndarray): Price time series
    ### Output:
        - Time series of returns (np.ndarray)
    """
    type_check(value=price_array, type_=np.ndarray, value_name="price_array")
    returns = np.diff(a=price_array)/price_array[1:]
    return np.insert(returns, 0, 0)



def returns_average(price_array: np.ndarray, method: ReturnsMethod, n_periods: int=252) -> float:
    """
    ## Description
    Calculate the average return of a price array.
    ### Input:
        - price_array (np.ndarray): Price time series
        - method (ReturnsMethod): Method for computing the returns
        - n_periods (int): Number of periods used to estimate the average over (e.g., for daily prices n_periods=252 produces annualized average)
    ### Output:
        - Average return (float) over a certain period
    """
    type_check(value=method, type_=ReturnsMethod, value_name="method")
    n_periods = int(n_periods)
    returns = prices_to_returns(price_array=price_array)
    match method:
        case ReturnsMethod.IMPLIED_AVERAGE_RETURN:
            mean_return = np.mean(returns)
            return ((1+mean_return)**n_periods)-1
        case ReturnsMethod.ESTIMATED_FROM_TOTAL_RETURN:
            compounded_return = (1 + returns).prod()
            return compounded_return**(n_periods/len(returns)) - 1



def returns_std(price_array: np.ndarray, n_periods: int=252) -> float:
    """
    ## Description
    Calculate the standard deviation of the returns of a price array.
    ### Input:
        - price_array (np.ndarray): Price time series
        - n_periods (int): Number of periods used to estimate the standard deviation over (e.g., for daily prices n_periods=252 produces annualized standard deviation)
    ### Output:
        - Standard deviation (float) of returns over a certain period
    """
    n_periods = int(n_periods)
    returns_std = np.std(a=prices_to_returns(price_array=price_array))
    return returns_std * np.sqrt(n_periods)



def returns_variance(price_array: np.ndarray, n_periods: int=252) -> float:
    """
    ## Description
    Calculate the variance of the returns of a price array.
    ### Input:
        - price_array (np.ndarray): Price time series
        - n_periods (int): Number of periods used to estimate the variance over (e.g., for daily prices n_periods=252 produces annualized variance)
    ### Output:
        - Variance (float) of returns over a certain period
    """
    n_periods = int(n_periods)
    returns_variance = np.std(a=prices_to_returns(price_array=price_array))**2
    return returns_variance * n_periods