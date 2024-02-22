from typing import Union
import abc
from dataclasses import dataclass
import numpy as np
from digifi.utilities.general_utils import (type_check, DataClassValidation)
from digifi.utilities.time_value_utils import (CompoundingType, Compounding)
from digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from digifi.portfolio_applications.general import PortfolioInstrumentStruct



@dataclass(slots=True)
class ForwardRateAgreementStruct(DataClassValidation):
    """
    ## Description
    Parameters for the ForwardRateAgreement class.
    ### Input:
        - agreed_fixed_ratev (float): Agreed fixed rate of the contract
        - current_forward_rate (float): Current market-derived forward rate of similar contracts
        - time_to_maturity (float): Time to maturity of the contract
        - principal (float): Principal of the forward rate agreement
        - initial_price (float): Initial price of the forward rate contract
        - compounding_type (CmpoundingType): Compounding type used to discount cashflows
    """
    agreed_fixed_rate: float
    current_forward_rate: float
    time_to_maturity: float
    principal: float
    initial_price: float = 0.0
    compounding_type: CompoundingType=CompoundingType.PERIODIC

    def validate_time_to_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The parameter time_to_maturity must be positive.")
        return value



class ForwardRateAgreementInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "rate_adjustment") and
                callable(subclass.rate_adjustment))
    
    @abc.abstractmethod
    def rate_adjustment(self, futures_rate: float, in_place: bool=False) -> float:
        """
        ## Description
        Calculate forward rate adjustment based on futures rate.
        """
        raise NotImplementedError



def forward_interest_rate(zero_rate_1: float, time_1: float, zero_rate_2: float, time_2: float) -> float:
    """
    ## Description
    Forward interest rate for the period between time_1 and time_2.
    ### Input:
        - zero_rate_1 (float): Zero rate at time step 1
        - time_1 (float): Time step 1
        - zero_rate_2 (float): Zero rate at time step 2
        - time_2 (float): Time step 2
    ### Output:
        - Forward rate (float) from zero rate at time step 1 to zero rate at time step 2
    ### LaTeX Formula:
        - R_{f} = \\frac{R_{2}T_{2} - R_{1}T_{1}}{T_{2} - T_{1}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
        - Original Source: N/A
    """
    time_1 = float(time_1)
    time_2 = float(time_2)
    return (float(zero_rate_2)*time_2 - float(zero_rate_1)*time_1) / (time_2 - time_1)



def future_zero_rate(zero_rate_1: float, time_1: float, time_2: float, forward_rate: float) -> float:
    """
    ## Description
    Zero rate defined through the previous zero rate and current forward rate.
    ### Input:
        - zero_rate_1 (float): Zero rate at time step 1
        - time_1 (float): Time step 1
        - time_2 (float): Time step 2
        - forward_rate (float): Current forward rate
    ### Output:
        - Zero rate (float) at time step 2
    ### LaTeX Formula:
        - R_{2} = \\frac{R_{F}(T_{2}-T_{1}) + R_{1}T_{1}}{T_{2}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
        - Original Source: N/A
    """
    time_1 = float(time_1)
    time_2 = float(time_2)
    return (float(forward_rate)*(time_2-time_1) + float(zero_rate_1)*time_1)/time_2



class ForwardRateAgreement(FinancialInstrumentInterface, ForwardRateAgreementInterface):
    """
    ## Description
    Forward rate agreement and its methods.
    ### Input:
        - forward_rate_agreement_struct (ForwardRateAgreementStruct): Parameters for defining a ForwardRayeAgreement instance
        - financial_instrument_struct (FinancialInstrumentStruct): Parameters for defining regulatory categorization of an instrument
        - portfolio_instrument_struct (PortfolioInstrumentStruct): Parameters for defining historical data for portfolio construction and applications
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate_agreement
        - Original Source: N/A
    """
    def __init__(self, forward_rate_agreement_struct: ForwardRateAgreementStruct,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.FOREIGN_EXCHANGE_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]),
                                                                                                  portfolio_predicatble_income=np.array([]))) -> None:
        # Arguments validation
        type_check(value=forward_rate_agreement_struct, type_=ForwardRateAgreementStruct, value_name="forward_rate_agreement_struct")
        type_check(value=financial_instrument_struct, type_=FinancialInstrumentStruct, value_name="financial_instrument_struct")
        type_check(value=portfolio_instrument_struct, type_=PortfolioInstrumentStruct, value_name="portfolio_instrument_struct")
        # ForwardRateAgreementStruct parameters
        self.agreed_fixed_rate = forward_rate_agreement_struct.agreed_fixed_rate
        self.current_forward_rate = forward_rate_agreement_struct.current_forward_rate
        self.time_to_maturity = forward_rate_agreement_struct.time_to_maturity
        self.principal = forward_rate_agreement_struct.principal
        self.initial_price = forward_rate_agreement_struct.initial_price
        self.compounding_type = forward_rate_agreement_struct.compounding_type
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_predictable_income = portfolio_instrument_struct.portfolio_predicatble_income
    
    def __str__(self):
        return f"Forward Rate Agreement: {self.identifier}"
    
    def __latest_forward_rate(self, current_forward_rate: Union[float, None]=None) -> float:
        """
        ## Description
        Latest forward rate of the contract.\n
        Helper method to update current_forward_rate during calculations.
        """
        if isinstance(current_forward_rate, type(None)) is False:
            current_forward_rate = float(current_forward_rate)
        else:
            current_forward_rate = self.initial_spot_price
        return current_forward_rate

    def __latest_time_to_maturity(self, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        ## Description
        Latest time to maturity.\n
        Helper method to update time_to_maturity during calculations.
        """
        if isinstance(current_time_to_maturity, type(None))==False:
            current_time_to_maturity = self.time_to_maturity - float(current_time_to_maturity)
        else:
            current_time_to_maturity = self.time_to_maturity
        return current_time_to_maturity
    
    def present_value(self, current_forward_rate: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        ## Description
        Present values of the forward rate agreement.
        ### Input:
            - current_forward_rate (Union[float, None]): Current market forward rate (If none given, the current_forward_rate from the instance definition will be used)
            - current_time_to_maturity Union[float, None]: Current time too maturity (If none given, the time_to_maturity from the instance definition will be used)
        ### Output:
            - Present value (float) of the forward rate agreement
        ### LaTeX Formula:
            - PV = \\tau(R_{F} - R_{K})L
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate_agreement#Valuation_and_pricing
            - Original Source: N/A
        """
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        current_forward_rate = self.__latest_forward_rate(current_forward_rate=current_forward_rate)
        discount_term = Compounding(rate=current_forward_rate, compounding_type=self.compounding_type, compounding_frequency=1)
        return (current_time_to_maturity * (current_forward_rate - self.agreed_fixed_rate) * self.principal) * discount_term.compounding_term(time=current_time_to_maturity)

    def net_present_value(self, current_forward_rate: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        ## Description
        Net present value of the forward rate agreement.
        ### Input:
            - current_forward_rate (Union[float, None]): Current market forward rate (If none given, the current_forward_rate from the instance definition will be used)
            - current_time_to_maturity (Union[float, None]): Current time too maturity (If none given, the time_to_maturity from the instance definition will be used)
        ### Output:
            - Present value of the forward rate agreement minus the initial price it took to purchase the contract (float)
        """
        return -self.initial_price + self.present_value(current_forward_rate=current_forward_rate,
                                                        current_time_to_maturity=current_time_to_maturity)
    
    def future_value(self, current_forward_rate: Union[float, None]=None, current_time_to_maturity: Union[float, None]=None) -> float:
        """
        ## Description
        Future value of the forward rate agreement.
        ### Input:
            - current_forward_rate (Union[float, None]): Current market forward rate (If none given, the current_forward_rate from the instance definition will be used)
            - current_time_to_maturity (Union[float, None]): Current time too maturity (If none given, the time_to_maturity from the instance definition will be used)
        ### Output:
            - Future value (float) of the forward rate agreement at it maturity (Computed from the present value of the forward rate agreement)
        """
        current_time_to_maturity = self.__latest_time_to_maturity(current_time_to_maturity=current_time_to_maturity)
        discount_term = Compounding(rate=current_forward_rate, compounding_type=self.compounding_type, compounding_frequency=1)
        return self.present_value(current_forward_rate=current_forward_rate, current_time_to_maturity=current_time_to_maturity)/discount_term.compounding_term(time=current_time_to_maturity)
    
    def forward_rate_from_zero_rates(self, zero_rate_1: float, time_1: float, zero_rate_2: float, time_2: float,
                                     in_place: bool=False) -> float:
        """
        ## Description
        Forward interest rate for the period between time_1 and time_2.
        ### Input:
            - zero_rate_1 (float): Zero rate at time step 1
            - time_1 (float): Time step 1
            - zero_rate_2 (float): Zero rate at time step 2
            - time_2 (float): Time step 2
            - in_place (bool): Overwrite current_forward_rate with the obtained forward rate
        ### Output:
            - Forward rate (float) from zero rate at time step 1 to zero rate at time step 2
        ### LaTeX Formula:
            - R_{f} = \\frac{R_{2}T_{2} - R_{1}T_{1}}{T_{2} - T_{1}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
            - Original Source: N/A
        """
        forward_rate = forward_interest_rate(zero_rate_1=zero_rate_1, time_1=time_1, zero_rate_2=zero_rate_2, time_2=time_2)
        if bool(in_place):
            self.current_forward_rate = forward_rate
        return forward_rate
    
    def zero_rate_from_forward_rate(self, zero_rate_1: float, time_1: float, time_2: float, current_forward_rate: Union[float, None]=None) -> float:
        """
        ## Description
        Zero rate defined through the previous zero rate and current forward rate.
        ### Input:
            - zero_rate_1 (float): Zero rate at time step 1
            - time_1 (float): Time step 1
            - time_2 (float): Time step 2
            - current_forward_rate (Union[float, None]): Current market forward rate (If none given, the current_forward_rate from the instance definition will be used)
        ### Output:
            - Zero rate (float) at time step 2
        ### LaTeX Formula:
            - R_{2} = \\frac{R_{F}(T_{2}-T_{1}) + R_{1}T_{1}}{T_{2}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
            - Original Source: N/A
        """
        current_forward_rate = self.__latest_forward_rate(current_forward_rate=current_forward_rate)
        return future_zero_rate(zero_rate_1=zero_rate_1, time_1=time_1, time_2=time_2, forward_rate=current_forward_rate)
    
    def rate_adjustment(self, futures_rate: float, convexity_adjustment: float, in_place: bool=False) -> float:
        """
        ## Description
        Adjustment of the forward rate based on futures rate.\n
        Forward Rate = Futures Rate - Convexity Adjustment.
        ### Input:
            - futures_rate (float): Current futures contract rate
            - convexity_sdjustment (float): Convexity adjustment constant
            - in_place (bool): Overwrite current_forward_rate with the obtained forward rate
        ### Output
            - Forward rate of the contract (float)
        ### LaTeX Formula:
            - \\textit{Forward Rate} = \\textit{Futures Rate} - \\textit{C}
        """
        convexity_adjustment = float(convexity_adjustment)
        if convexity_adjustment<=0:
            raise ValueError("The argument convexity adjustment must be positive.")
        forward_rate = float(futures_rate) - convexity_adjustment
        if bool(in_place):
            self.current_forward_rate = forward_rate
        return forward_rate