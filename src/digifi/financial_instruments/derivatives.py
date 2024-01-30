from typing import Union
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import (type_check, DataClassValidation)
from src.digifi.utilities.time_value_utils import (CompoundingType, Compounding)
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from src.digifi.portfolio_applications.general import PortfolioInstrumentStruct
from src.digifi.probability_distributions.continuous_probability_distributions import NormalDistribution
from src.digifi.lattice_based_models.general import LatticeModelPayoffType
from src.digifi.lattice_based_models.binomial_models import BrownianMotionBinomialModel
from src.digifi.lattice_based_models.trinomial_models import BrownianMotionTrinomialModel




class ContractType(Enum):
    FORWARD = 1
    FUTURES = 2



class OptionType(Enum):
    EUROPEAN = 1
    AMERICAN = 2
    BERMUDAN = 3



class OptionPayoffType(Enum):
    CALL = 1
    PUT = 2



class OptionPricingMethod(Enum):
    BLACK_SCHOLES = 1
    BINOMIAL = 2
    TRINOMIAL = 3



@dataclass(slots=True)
class FuturesContractStruct(DataClassValidation):
    contract_type: ContractType
    contract_price: float
    delivery_price: float
    discount_rate: float
    maturity: float
    initial_spot_price: float

    def validate_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The parameter maturity must be positive.")
        return value



@dataclass(slots=True)
class OptionStruct(DataClassValidation):
    asset_price: float
    strike_price: float
    discount_rate: float
    dividend_yield: float
    time_to_maturity: float
    sigma: float
    initial_option_price: float
    option_type: OptionType
    payoff_type: OptionPayoffType

    def validate_time_to_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The parameter time_to_maturity must be positive.")
        return value



class FuturesContractInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "append_predictable_yield") and
                callable(subclass.append_predictable_yield) and
                hasattr(subclass, "append_predictable_income") and
                callable(subclass.append_predictable_income) and
                hasattr(subclass, "forward_price") and
                callable(subclass.forward_price))
    
    @abc.abstractmethod
    def append_predictable_yield(self) -> None:
        """
        Append predictable yield to discount rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def append_predictable_income(self) -> None:
        """
        Append predictable income to initial spot price. 
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_price(self) -> float:
        """
        Calculate initial forward price.
        """
        raise NotImplementedError



class OptionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "call_payoff") and
                callable(subclass.call_payoff) and
                hasattr(subclass, "put_payoff") and
                callable(subclass.put_payoff) and
                hasattr(subclass, "delta") and
                callable(subclass.delta) and
                hasattr(subclass, "vega") and
                callable(subclass.vega) and
                hasattr(subclass, "theta") and
                callable(subclass.theta) and
                hasattr(subclass, "rho") and
                callable(subclass.rho) and
                hasattr(subclass, "epsilon") and
                callable(subclass.epsilon) and
                hasattr(subclass, "gamma") and
                callable(subclass.gamma))
    
    @staticmethod
    @abc.abstractmethod
    def call_payoff() -> Union[np.ndarray, float]:
        """
        Call payoff function.
        """
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def put_payoff() -> Union[np.ndarray, float]:
        """
        Put payoff function.
        """
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def delta() -> float:
        """
        Option delta.
        """
        return NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def vega() -> float:
        """
        Option vega.
        """
        return NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def theta() -> float:
        """
        Option theta.
        """
        return NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def rho() -> float:
        """
        Option rho.
        """
        return NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def epsilon() -> float:
        """
        Option epsilon.
        """
        return NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def gamma() -> float:
        """
        Option gamma.
        """
        return NotImplementedError



class FuturesContract(FinancialInstrumentInterface, FuturesContractInterface):
    """
    Futures contract financial instrument and its methods.
    Can act as a parent class in a definition of 'Forward Contract' class.
    """
    def __init__(self, futures_contract_struct: FuturesContractStruct,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([])),
                 compounding_type: CompoundingType=CompoundingType.PERIODIC, compounding_frequency: int=1) -> None:
        # Arguments validation
        type_check(value=compounding_type, type_=CompoundingType, value_name="compounding_type")
        # FuturesContract class parameters
        self.compounding_type = compounding_type
        self.compounding_frequency = int(compounding_frequency)
        # FuturesContractStruct parameters
        self.contract_type = futures_contract_struct.contract_type
        self.contract_price = futures_contract_struct.contract_price
        self.delivery_price = futures_contract_struct.delivery_price
        self.discount_rate = futures_contract_struct.discount_rate
        self.maturity = futures_contract_struct.maturity
        self.initial_spot_price = futures_contract_struct.initial_spot_price
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
    
    def __str__(self):
        return f"Futures Contract: {self.identifier}"
    
    def __latest_spot_price(self, current_price: Union[float, None]=None) -> float:
        """
        Latest spot price of the contract.
        Helper method to update spot price during calculations.
        """
        if isinstance(current_price, type(None)) is False:
            current_price = float(current_price)
        else:
            current_price = self.initial_spot_price
        return current_price

    def __latest_time_to_maturity(self, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        Latest time to maturity.
        Helper method to update time to maturity during calculations.
        """
        if isinstance(current_time_to_maturity, type(None))==False:
            current_time_to_maturity = self.maturity - float(current_time_to_maturity)
        else:
            current_time_to_maturity = self.maturity
        return current_time_to_maturity
    
    def append_predictable_yield(self, yield_: float) -> None:
        self.discount_rate = self.discount_rate + float(yield_)
    
    def append_predictable_income(self, income: float) -> None:
        self.initial_spot_price = self.initial_spot_price + float(income)

    def forward_price(self, current_price: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        F_{0} = S_{0}e^{rT}
        """
        current_price = self.__latest_spot_price(current_price=current_price)
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        discount_term = Compounding(rate=self.discount_rate, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
        return current_price/discount_term.compounding_term(time=current_time_to_maturity)
    
    def present_value(self, current_price: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        PV = (F_{0} - K)e^{-rT}
        """
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        discount_term = Compounding(rate=self.discount_rate, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
        return (self.forward_price(current_price=current_price, current_time_to_maturity=current_time_to_maturity)-self.delivery_price)*discount_term.compounding_term(time=current_time_to_maturity)
    
    def net_present_value(self, current_price: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        return -self.contract_price + self.present_value(current_price=current_price, current_time_to_maturity=current_time_to_maturity)
    
    def future_value(self, current_price: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        discount_term = Compounding(rate=self.discount_rate, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
        return self.present_value(current_price=current_price, current_time_to_maturity=current_time_to_maturity)/discount_term.compounding_term(time=current_time_to_maturity)



class Option(FinancialInstrumentInterface, OptionInterface):
    """
    Option financial instrument and its methods.
    """
    def __init__(self, option_struct: OptionStruct, option_pricing_method: OptionPricingMethod=OptionPricingMethod.BINOMIAL,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]))) -> None:
        # Arguments validation
        type_check(value=option_pricing_method, type_=OptionPricingMethod, value_name="option_pricing_method")
        if (option_pricing_method==OptionPricingMethod.BLACK_SCHOLES) and (option_struct.option_type!=OptionType.EUROPEAN):
            raise ValueError("Black-Scholes option pricing is only available for European options.")
        # Option class parameters
        self.option_pricing_method = option_pricing_method
        # OptionStruct parameters
        self.asset_price = option_struct.asset_price
        self.strike_price = option_struct.strike_price
        self.discount_rate = option_struct.discount_rate
        self.dividend_yield = option_struct.dividend_yield
        self.time_to_maturity = option_struct.time_to_maturity
        self.sigma = option_struct.sigma
        self.initial_option_price = option_struct.initial_option_price
        self.option_type = option_struct.option_type
        self.payoff_type = option_struct.payoff_type
        match self.payoff_type:
            case OptionPayoffType.CALL:
                self.payoff = self.call_payoff
            case OptionPayoffType.PUT:
                self.payoff = self.put_payoff
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
    
    def __str__(self):
        return f"Option: {self.identifier}"
    
    def __is_european(self, method_name: str) -> None:
        """
        Check that the instance is a European option.
        """
        if self.option_type!=OptionType.EUROPEAN:
            raise ValueError("To use {} method it is required for the option_type argument to be defined as OptionType.EUROPEAN.".format(str(method_name)))
    
    @staticmethod
    def call_payoff(s_t: Union[np.ndarray, float], k: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Payoff = max(s_{t} - k, 0)
        """
        return np.maximum(s_t-k, 0)
    
    @staticmethod
    def put_payoff(s_t: Union[np.ndarray, float], k: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Payoff = max(k - s_{t}, 0)
        """
        return np.maximum(k - s_t, 0)
    
    @staticmethod
    def delta(option_value_1: float, option_value_0: float, asset_price_1: float, asset_price_0: float) -> float:
        """
        \Delta = \\frac{\partial V}{\partial S}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(asset_price_1) - float(asset_price_0))
    
    @staticmethod
    def vega(option_value_1: float, option_value_0: float, sigma_1: float, sigma_0: float) -> float:
        """
        \\nu = \\frac{\partial V}{\partial\sigma}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(sigma_1) - float(sigma_0))
    
    @staticmethod
    def theta(option_value_1: float, option_value_0: float, tau_1: float, tau_0: float) -> float:
        """
        \Theta = \\frac{\partial V}{\partial\\tau}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(tau_1) - float(tau_0))
    
    @staticmethod
    def rho(option_value_1: float, option_value_0: float, r_1: float, r_0: float) -> float:
        """
        \\rho = \\frac{\partial V}{\partial r}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(r_1) - float(r_0))
    
    @staticmethod
    def epsilon(option_value_1: float, option_value_0: float, dividend_1: float, dividend_0: float) -> float:
        """
        \epsilon = \\frac{\partial V}{\partial q}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(dividend_1) - float(dividend_0))

    @staticmethod
    def gamma(option_value_2: float, option_value_1: float, option_value_0: float, asset_price_2: float, asset_price_1: float,
              asset_price_0: float) -> float:
        """
        \Gamma = \\frac{\partial\Delat}{\partial S}.
        """
        delta_0 = (float(option_value_1) - float(option_value_0)) / (float(asset_price_1) - float(asset_price_0))
        delta_1 = (float(option_value_2) - float(option_value_1)) / (float(asset_price_2) - float(asset_price_1))
        return (delta_1 - delta_0) / (float(asset_price_2) - float(asset_price_0))
    
    def __european_option_black_scholes_params(self) -> dict:
        """
        Parameters d_1 and d_2 for European option.
        """
        d_1 = (np.log(self.asset_price/self.strike_price) + (self.discount_rate-self.dividend_yield+0.5*self.sigma**2)*self.time_to_maturity) / (self.sigma*np.sqrt(self.time_to_maturity))
        d_2 = d_1 - self.sigma*np.sqrt(self.time_to_maturity)
        return {"d_1":d_1, "d_2":d_2}
    
    def __european_option_black_scholes_value(self) -> float:
        """
        Value of the European option evaluated using Black-Scholes formula.
        """
        self.__is_european(method_name="Black-Scholes pricing")
        black_scholes_params = self.__european_option_black_scholes_params()
        d_1, d_2 = black_scholes_params["d_1"], black_scholes_params["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.CALL:
                return (self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1) -
                        self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2))
            case OptionPayoffType.PUT:
                return (self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2) -
                        self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1))
    
    def european_option_delta(self) -> float:
        self.__is_european(method_name="european_option_delta")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.CALL:
                return np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)
            case OptionPayoffType.PUT:
                return -np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1)
    
    def european_option_vega(self) -> float:
        self.__is_european(method_name="european_option_vega")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        return self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)*np.sqrt(self.time_to_maturity)
    
    def european_option_theta(self) -> float:
        self.__is_european(method_name="european_option_theta")
        black_scholes_params = self.__european_option_black_scholes_params()
        d_1, d_2 = black_scholes_params["d_1"], black_scholes_params["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.CALL:
                return (-np.exp(-self.dividend_yield*self.time_to_maturity)*self.asset_price*normal_dist.cdf(x=d_1)*self.sigma/(2*np.sqrt(self.time_to_maturity)) -
                        self.discount_rate*self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2) +
                        self.dividend_yield*self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1))
            case OptionPayoffType.PUT:
                return (-np.exp(-self.dividend_yield*self.time_to_maturity)*self.asset_price*normal_dist.cdf(x=d_1)*self.sigma/(2*np.sqrt(self.time_to_maturity)) +
                        self.discount_rate*self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2) -
                        self.dividend_yield*self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1))
    
    def european_option_rho(self) -> float:
        self.__is_european(method_name="european_option_rho")
        d_2 = self.__european_option_black_scholes_params()["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.CALL:
                return self.strike_price*self.time_to_maturity*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2)
            case OptionPayoffType.PUT:
                return -self.strike_price*self.time_to_maturity*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2)

    def european_option_epsilon(self) -> float:
        self.__is_european(method_name="european_option_epsilon")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.CALL:
                return -self.asset_price*self.time_to_maturity*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)
            case OptionPayoffType.PUT:
                return self.asset_price*self.time_to_maturity*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1)

    def european_option_gamma(self) -> float:
        self.__is_european(method_name="european_option_gamma")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        return np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)/(self.asset_price*self.sigma*np.sqrt(self.time_to_maturity))

    def present_value(self, lattice_model_n_steps: int=500) -> float:
        # Black-Scholes solution only is applicable to European options by constructor limitation
        if self.option_pricing_method==OptionPricingMethod.BLACK_SCHOLES:
            return self.__european_option_black_scholes_value()
        # Translation of OptionPayoffType to LatticeModelPayoffType
        match self.payoff_type:
            case OptionPayoffType.CALL:
                lattice_model_payoff_type = LatticeModelPayoffType.CALL
            case OptionPayoffType.PUT:
                lattice_model_payoff_type = LatticeModelPayoffType.PUT
        # Selection of lattice model
        match self.option_pricing_method:
            case OptionPricingMethod.BINOMIAL:
                lattice_model = BrownianMotionBinomialModel(s_0=self.asset_price, k=self.strike_price, T=self.time_to_maturity,
                                                            r=self.discount_rate, sigma=self.sigma, q=self.dividend_yield,
                                                            n_steps=int(lattice_model_n_steps), payoff_type=lattice_model_payoff_type)
            case OptionPricingMethod.TRINOMIAL:
                lattice_model = BrownianMotionTrinomialModel(s_0=self.asset_price, k=self.strike_price, T=self.time_to_maturity,
                                                             r=self.discount_rate, sigma=self.sigma, q=self.dividend_yield,
                                                             n_steps=int(lattice_model_n_steps), payoff_type=lattice_model_payoff_type)
        # Selection of option type
        match self.option_type:
            case OptionType.EUROPEAN:
                return lattice_model.european_option()
            case OptionType.AMERICAN:
                return lattice_model.american_option()
            case OptionType.BERMUDAN:
                return lattice_model.bermudan_option()
    
    def net_present_value(self) -> float:
        return -self.initial_option_price + self.present_value()
    
    def future_value(self) -> float:
        return self.present_value()*np.exp(self.discount_rate*self.time_to_maturity)
    
    def present_value_surface(self, start_price: float, stop_price: float, n_prices: int=100, n_timesteps: int=100,
                              lattice_model_n_steps: int=100) -> dict:
        lattice_model_n_steps = int(lattice_model_n_steps)
        times_to_maturity = np.linspace(start=self.time_to_maturity, stop=0, num=int(n_timesteps))
        price_array = np.linspace(start=float(start_price), stop=float(stop_price), num=int(n_prices))
        option_pv_matrix = []
        for i in range(n_timesteps):
            option_pv_array = np.array([])
            if i==n_timesteps-1:
                option_pv_array = np.append(option_pv_array, self.payoff(s_t=price_array, k=self.strike_price))
            else:
                self.time_to_maturity = times_to_maturity[i]
                for j in range(n_timesteps):
                    self.asset_price = price_array[j]
                    option_pv_array = np.append(option_pv_array, self.present_value(lattice_model_n_steps=lattice_model_n_steps))
            option_pv_matrix.append(option_pv_array)
        return {"times_to_maturity":times_to_maturity, "price_array":price_array, "option_pv_matrix":np.array(option_pv_matrix)}