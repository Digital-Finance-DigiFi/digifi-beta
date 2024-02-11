from typing import (Union, Type)
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import (type_check, DataClassValidation)
from src.digifi.utilities.time_value_utils import (CompoundingType, Compounding)
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from src.digifi.financial_instruments.derivatives_utils import (CustomPayoff, LongCallPayoff, LongPutPayoff, validate_custom_payoff)
from src.digifi.portfolio_applications.general import PortfolioInstrumentStruct
from src.digifi.probability_distributions.continuous_probability_distributions import NormalDistribution
from src.digifi.lattice_based_models.general import LatticeModelPayoffType
from src.digifi.lattice_based_models.binomial_models import BrownianMotionBinomialModel
from src.digifi.lattice_based_models.trinomial_models import BrownianMotionTrinomialModel




class ContractType(Enum):
    FORWARD = 1
    FUTURES = 2
    BILLS_OF_EXCHANGE = 3



class OptionType(Enum):
    EUROPEAN = 1
    AMERICAN = 2
    BERMUDAN = 3



class OptionPayoffType(Enum):
    LONG_CALL = 1
    LONG_PUT = 2
    CUSTOM = 3



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
    initial_spot_price: float = 0.0
    compounding_type: CompoundingType = CompoundingType.PERIODIC
    compounding_frequency: int = 1

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
    option_type: OptionType
    payoff_type: OptionPayoffType
    initial_option_price: float = 0.0

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
        return (hasattr(subclass, "delta") and
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



def minimum_variance_hedge_ratio(asset_price_sigma: float, contract_price_sigma: float, asset_to_contract_corr: float) -> float:
        """
        ## Description
        Mininum-variance hedge ratio for a forward contract that hedges an underlying asset.\n
        Note: this assumes that there is a linear relationship between the asset and the contract.
        ### LaTeX Formula:
            - h^{*} = \\rho\\frac{\\sigma_{S}}{\\sigma_{F}}
        """
        return float(asset_to_contract_corr)*float(asset_price_sigma)/float(contract_price_sigma)



class FuturesContract(FinancialInstrumentInterface, FuturesContractInterface):
    """
    ## Description
    Futures contract financial instrument and its methods.\n
    Can act as a parent class in a definition of 'Forward Contract' or 'Bill of Exchange' classes.
    ## Links
    - Wikipedia: https://en.wikipedia.org/wiki/Futures_contract
    - Original Source: N/A
    """
    def __init__(self, futures_contract_struct: FuturesContractStruct,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]),
                                                                                                  portfolio_predicatble_income=np.array([]))) -> None:
        # Arguments validation
        type_check(value=futures_contract_struct, type_=FuturesContractStruct, value_name="futures_contract_struct")
        type_check(value=financial_instrument_struct, type_=FinancialInstrumentStruct, value_name="financial_instrument_struct")
        type_check(value=portfolio_instrument_struct, type_=PortfolioInstrumentStruct, value_name="portfolio_instrument_struct")
        # FuturesContractStruct parameters
        self.contract_type = futures_contract_struct.contract_type
        self.contract_price = futures_contract_struct.contract_price
        self.delivery_price = futures_contract_struct.delivery_price
        self.discount_rate = futures_contract_struct.discount_rate
        self.maturity = futures_contract_struct.maturity
        self.initial_spot_price = futures_contract_struct.initial_spot_price
        self.compounding_type = futures_contract_struct.compounding_type
        self.compounding_frequency = futures_contract_struct.compounding_frequency
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_predictable_income = portfolio_instrument_struct.portfolio_predicatble_income
    
    def __str__(self):
        return f"Futures Contract: {self.identifier}"
    
    def __latest_spot_price(self, current_price: Union[float, None]=None) -> float:
        """
        ## Description
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
        ## Description
        Latest time to maturity.
        Helper method to update time_to_maturity during calculations.
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
        ## Description
        F_{t} = S_{t}e^{r\\tau}
        """
        current_price = self.__latest_spot_price(current_price=current_price)
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        discount_term = Compounding(rate=self.discount_rate, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
        return current_price/discount_term.compounding_term(time=current_time_to_maturity)
    
    def present_value(self, current_price: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        ## Description
        PV = (F_{t} - K)e^{-r\\tau}
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
    ## Description
    Option financial instrument and its methods.
    ## Links
    - Wikipedia: https://en.wikipedia.org/wiki/Option_(finance)
    - Original Source: N/A
    """
    def __init__(self, option_struct: OptionStruct, option_pricing_method: OptionPricingMethod=OptionPricingMethod.BINOMIAL,
                 payoff: Union[Type[CustomPayoff], None]=None,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]),
                                                                                                  portfolio_predicatble_income=np.array([]))) -> None:
        # Arguments validation
        type_check(value=option_struct, type_=OptionStruct, value_name="option_struct")
        type_check(value=option_pricing_method, type_=OptionPricingMethod, value_name="option_pricing_method")
        type_check(value=financial_instrument_struct, type_=FinancialInstrumentStruct, value_name="financial_instrument_struct")
        type_check(value=portfolio_instrument_struct, type_=PortfolioInstrumentStruct, value_name="portfolio_instrument_struct")
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
        self.option_type = option_struct.option_type
        self.payoff_type = option_struct.payoff_type
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                self.payoff = self.__long_call_payoff()
            case OptionPayoffType.LONG_PUT:
                self.payoff = self.__long_put_payoff()
            case OptionPayoffType.CUSTOM:
                if isinstance(payoff, CustomPayoff):
                    self.payoff = validate_custom_payoff(custom_payoff=payoff)
                else:
                    raise ValueError("For the CUSTOM payoff type, the argument custom_payoff must be defined.")
        self.initial_option_price = option_struct.initial_option_price
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_predictable_income = portfolio_instrument_struct.portfolio_predicatble_income
    
    def __str__(self):
        return f"Option: {self.identifier}"
    
    def __is_european(self, method_name: str) -> None:
        """
        ## Description
        Check that the instance is a European option.
        """
        if self.option_type!=OptionType.EUROPEAN:
            raise ValueError("To use {} method it is required for the option_type argument to be defined as OptionType.EUROPEAN.".format(str(method_name)))

    def __is_custom_payoff(self, method_name: str) -> None:
        """
        ## Description
        Check that the instance does not have the custom payoff.
        """
        if self.payoff_type==OptionPayoffType.CUSTOM:
            raise ValueError("To use {} method is is required for the payoff_type argument not to be CUSTOM.",format(str(method_name)))
    
    def __long_call_payoff(self) -> Type[CustomPayoff]:
        """
        ## Description
        Payoff = max(s_{t} - k, 0)
        """
        return LongCallPayoff(k=self.strike_price)
    
    def __long_put_payoff(self) -> Type[CustomPayoff]:
        """
        ## Description
        Payoff = max(k - s_{t}, 0)
        """
        return LongPutPayoff(k=self.strike_price)
    
    @staticmethod
    def delta(option_value_1: float, option_value_0: float, asset_price_1: float, asset_price_0: float) -> float:
        """
        ## Description
        \\Delta = \\frac{\\partial V}{\\partial S}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(asset_price_1) - float(asset_price_0))
    
    @staticmethod
    def vega(option_value_1: float, option_value_0: float, sigma_1: float, sigma_0: float) -> float:
        """
        ## Description
        \\nu = \\frac{\partial V}{\\partial\\sigma}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(sigma_1) - float(sigma_0))
    
    @staticmethod
    def theta(option_value_1: float, option_value_0: float, tau_1: float, tau_0: float) -> float:
        """
        ## Description
        \\Theta = \\frac{\\partial V}{\\partial\\tau}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(tau_1) - float(tau_0))
    
    @staticmethod
    def rho(option_value_1: float, option_value_0: float, r_1: float, r_0: float) -> float:
        """
        ## Description
        \\rho = \\frac{\\partial V}{\\partial r}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(r_1) - float(r_0))
    
    @staticmethod
    def epsilon(option_value_1: float, option_value_0: float, dividend_1: float, dividend_0: float) -> float:
        """
        ## Description
        \\epsilon = \\frac{\\partial V}{\\partial q}.
        """
        return (float(option_value_1) - float(option_value_0)) / (float(dividend_1) - float(dividend_0))

    @staticmethod
    def gamma(option_value_2: float, option_value_1: float, option_value_0: float, asset_price_2: float, asset_price_1: float,
              asset_price_0: float) -> float:
        """
        ## Description
        \\Gamma = \\frac{\\partial\\Delat}{\\partial S}.
        """
        delta_0 = (float(option_value_1) - float(option_value_0)) / (float(asset_price_1) - float(asset_price_0))
        delta_1 = (float(option_value_2) - float(option_value_1)) / (float(asset_price_2) - float(asset_price_1))
        return (delta_1 - delta_0) / (float(asset_price_2) - float(asset_price_0))
    
    def __european_option_black_scholes_params(self) -> dict:
        """
        ## Description
        Parameters d_1 and d_2 for the European option.
        """
        self.__is_european(method_name="__european_option_black_scholes_params")
        d_1 = (np.log(self.asset_price/self.strike_price) + (self.discount_rate-self.dividend_yield+0.5*self.sigma**2)*self.time_to_maturity) / (self.sigma*np.sqrt(self.time_to_maturity))
        d_2 = d_1 - self.sigma*np.sqrt(self.time_to_maturity)
        return {"d_1":d_1, "d_2":d_2}
    
    def __european_option_black_scholes_value(self) -> float:
        """
        ## Description
        Value of the European option evaluated using Black-Scholes formula.
        """
        self.__is_european(method_name="Black-Scholes pricing")
        self.__is_custom_payoff(method_name="Black-Scholes pricing")
        black_scholes_params = self.__european_option_black_scholes_params()
        d_1, d_2 = black_scholes_params["d_1"], black_scholes_params["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                return (self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1) -
                        self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2))
            case OptionPayoffType.LONG_PUT:
                return (self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2) -
                        self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1))
            case OptionPayoffType.CUSTOM:
                # Should never be accessed
                return None
    
    def european_option_delta(self) -> float:
        self.__is_european(method_name="european_option_delta")
        self.__is_custom_payoff(method_name="european_option_delta")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                return np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)
            case OptionPayoffType.LONG_PUT:
                return -np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1)
            case OptionPayoffType.CUSTOM:
                # Should never be accessed
                return None
    
    def european_option_vega(self) -> float:
        self.__is_european(method_name="european_option_vega")
        self.__is_custom_payoff(method_name="european_option_vega")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        return self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)*np.sqrt(self.time_to_maturity)
    
    def european_option_theta(self) -> float:
        self.__is_european(method_name="european_option_theta")
        self.__is_custom_payoff(method_name="european_option_theta")
        black_scholes_params = self.__european_option_black_scholes_params()
        d_1, d_2 = black_scholes_params["d_1"], black_scholes_params["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                return (-np.exp(-self.dividend_yield*self.time_to_maturity)*self.asset_price*normal_dist.cdf(x=d_1)*self.sigma/(2*np.sqrt(self.time_to_maturity)) -
                        self.discount_rate*self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2) +
                        self.dividend_yield*self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1))
            case OptionPayoffType.LONG_PUT:
                return (-np.exp(-self.dividend_yield*self.time_to_maturity)*self.asset_price*normal_dist.cdf(x=d_1)*self.sigma/(2*np.sqrt(self.time_to_maturity)) +
                        self.discount_rate*self.strike_price*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2) -
                        self.dividend_yield*self.asset_price*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1))
            case OptionPayoffType.CUSTOM:
                # Should never be accessed
                return None
    
    def european_option_rho(self) -> float:
        self.__is_european(method_name="european_option_rho")
        self.__is_custom_payoff(method_name="european_option_rho")
        d_2 = self.__european_option_black_scholes_params()["d_2"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                return self.strike_price*self.time_to_maturity*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=d_2)
            case OptionPayoffType.LONG_PUT:
                return -self.strike_price*self.time_to_maturity*np.exp(-self.discount_rate*self.time_to_maturity)*normal_dist.cdf(x=-d_2)
            case OptionPayoffType.CUSTOM:
                # Should never be accessed
                return None

    def european_option_epsilon(self) -> float:
        self.__is_european(method_name="european_option_epsilon")
        self.__is_custom_payoff(method_name="european_option_epsilon")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                return -self.asset_price*self.time_to_maturity*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)
            case OptionPayoffType.LONG_PUT:
                return self.asset_price*self.time_to_maturity*np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=-d_1)
            case OptionPayoffType.CUSTOM:
                # Should never be accessed
                return None

    def european_option_gamma(self) -> float:
        self.__is_european(method_name="european_option_gamma")
        self.__is_custom_payoff(method_name="european_option_gamma")
        d_1 = self.__european_option_black_scholes_params()["d_1"]
        normal_dist = NormalDistribution(mu=0, sigma=1)
        return np.exp(-self.dividend_yield*self.time_to_maturity)*normal_dist.cdf(x=d_1)/(self.asset_price*self.sigma*np.sqrt(self.time_to_maturity))

    def present_value(self, lattice_model_n_steps: int=500) -> float:
        # Black-Scholes solution only is applicable to European options by constructor limitation
        if self.option_pricing_method==OptionPricingMethod.BLACK_SCHOLES:
            return self.__european_option_black_scholes_value()
        # Translation of OptionPayoffType to LatticeModelPayoffType
        match self.payoff_type:
            case OptionPayoffType.LONG_CALL:
                lattice_model_payoff_type = LatticeModelPayoffType.LONG_CALL
            case OptionPayoffType.LONG_PUT:
                lattice_model_payoff_type = LatticeModelPayoffType.LONG_PUT
            case OptionPayoffType.CUSTOM:
                lattice_model_payoff_type = LatticeModelPayoffType.CUSTOM
        # Selection of lattice model
        match self.option_pricing_method:
            case OptionPricingMethod.BINOMIAL:
                lattice_model = BrownianMotionBinomialModel(s_0=self.asset_price, k=self.strike_price, T=self.time_to_maturity,
                                                            r=self.discount_rate, sigma=self.sigma, q=self.dividend_yield,
                                                            n_steps=int(lattice_model_n_steps), payoff_type=lattice_model_payoff_type,
                                                            custom_payoff=self.payoff)
            case OptionPricingMethod.TRINOMIAL:
                lattice_model = BrownianMotionTrinomialModel(s_0=self.asset_price, k=self.strike_price, T=self.time_to_maturity,
                                                             r=self.discount_rate, sigma=self.sigma, q=self.dividend_yield,
                                                             n_steps=int(lattice_model_n_steps), payoff_type=lattice_model_payoff_type,
                                                             custom_payoff=self.payoff)
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
        n_prices = int(n_prices)
        lattice_model_n_steps = int(lattice_model_n_steps)
        times_to_maturity = np.linspace(start=self.time_to_maturity, stop=0, num=int(n_timesteps))
        price_array = np.linspace(start=float(start_price), stop=float(stop_price), num=n_prices)
        option_pv_matrix = []
        for i in range(n_timesteps):
            option_pv_array = np.array([])
            if i==n_timesteps-1:
                option_pv_array = np.append(option_pv_array, self.payoff.payoff(s_t=price_array))
            else:
                self.time_to_maturity = times_to_maturity[i]
                for j in range(n_prices):
                    self.asset_price = price_array[j]
                    option_pv_array = np.append(option_pv_array, self.present_value(lattice_model_n_steps=lattice_model_n_steps))
            option_pv_matrix.append(option_pv_array)
        return {"times_to_maturity":times_to_maturity, "price_array":price_array, "option_pv_matrix":np.array(option_pv_matrix)}