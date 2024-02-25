from typing import Union
import copy
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
from digifi.utilities.general_utils import (compare_array_len, type_check, DataClassValidation)
from digifi.utilities.time_value_utils import (Cashflow, CompoundingType, Compounding, internal_rate_of_return)
from digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from digifi.portfolio_applications.general import PortfolioInstrumentStruct



class BondType(Enum):
    ANNUITY_BOND = 1
    GROWING_ANNUITY_BOND = 2
    ZERO_COUPON_BOND = 3
    CUSTOM = 4
    # TODO: Add convertible, callable and puttable bonds



@dataclass(slots=True)
class BondStruct(DataClassValidation):
    """
    ## Description
    Parameters for the Bond class.
    ### Input:
        - bond_type (BondType): Type of bond (i.e., ANNUITY_BOND (Coupon bond), GROWING_ANNUITY_BOND (Coupon bond with growing coupon), ZERO_COUPON_BOND, or CUSTOM)
        - principal (float): Principal of the bond
        - coupon_rate (float): Coupon rate of the bond (Initial coupon rate for the GROWING_ANNUITY_BOND)
        - discount_rate (Union[np.ndarray, float]): Discount rate(s) for discounting future coupons
        - maturity (float): Maturity of the bond
        - initial_price (float): Price at which the bond was initially purchased
        - compounding_type (CompoundingType): Compounding type used to discount cashflows
        - compounding_frequency (int): Compounding frequency of bond's payoff (e.g., for a bond with 3% semi-annual coupon rate, the compounding frequency is 2)
        - coupon_growth_rate (float): Growth rate of the coupon
        - inflation_rate (float): Inflation rate with which the cashflows will be discounted (Applies multiplicatively on top of discoun_rates)
        - first_coupon_time (float): Time at which the initial coupon is scheduled to be paid
    """
    bond_type: BondType
    principal: float
    coupon_rate: float
    discount_rate: Union[np.ndarray, float]
    maturity: float
    initial_price: float = 0.0
    compounding_type: CompoundingType = CompoundingType.PERIODIC
    compounding_frequency: int = 1
    coupon_growth_rate: float = 0.0
    inflation_rate: float = 0.0
    first_coupon_time: float = 1.0

    def validate_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The parameter maturity must be positive.")
        return value



class BondInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "yield_to_maturity") and
                callable(subclass.yield_to_maturity) and
                hasattr(subclass, "zero_rate") and
                callable(subclass.zero_rate) and
                hasattr(subclass, "par_yield") and
                callable(subclass.par_yield) and
                hasattr(subclass, "duration") and
                callable(subclass.duration) and
                hasattr(subclass, "convexity") and
                callable(subclass.convexity) and
                hasattr(subclass, "accrued_interest") and
                callable(subclass.accrued_interest) and
                hasattr(subclass, "hazard_rate") and
                callable(subclass.hazard_rate))
    
    @abc.abstractmethod
    def yield_to_maturity(self) -> float:
        """
        ## Description
        Calculate yield to maturity.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def spot_rate(self) -> float:
        """
        ## Description
        Calculate spot rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def par_yield(self) -> float:
        """
        ## Description
        Calculate par yield.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def duration(self) -> float:
        """
        ## Description
        Calculate duration.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def convexity(self) -> float:
        """
        ## Description
        Calculate convexity.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def accrued_interest(self) -> float:
        """
        ## Description
        Calculate accrued interest.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def hazard_rate(self) -> float:
        """
        ## Description
        Calculate hazard rate.
        """
        raise NotImplementedError



def bond_price_from_yield(current_price: float, duration: float, convexity: float, yield_spread: float) -> float:
    """
    ## Description
    Bond pricing via Taylor series expansion of bond price assuming it only depends on the yield.
    ### Input:
        - current_price (float): Current market price of the bond
        - duration (float): Duration of the bond (i.e., first derivative of bond price with respect to yield)
        - convexity (float): Convexity of the bond (i.e., second derivative of bond price with respect to yield)
        - yield_spread (float): Expected change in bond's yield between current and future time step
    ### Output:
        - Price of the bond (float) in the next time step
    ### LaTeX Formula:
        - B_{t}-B_{t-1} = \Delta B_{t} = \\frac{dB}{dy}\\Delta y + 0.5\\frac{d^{2}B}{dy^{2}}\Delta y^{2}
    """
    current_price = float(current_price)
    yield_spread = float(yield_spread)
    db_dy = -current_price*float(duration)
    d2b_dy2 = float(convexity)*current_price
    future_bond_price = current_price + db_dy*(yield_spread) + (1/2)*d2b_dy2*(yield_spread)**2
    return future_bond_price



def bootstrap(principals: np.ndarray, maturities: np.ndarray, coupons: np.ndarray, prices: np.ndarray, coupon_dt: np.ndarray) -> np.ndarray:
    """
    ## Description
    Spot rate computation for the given list of bonds.
    ### Input:
        - principals (np.ndarray): Array of bond principals
        - maturities (np.ndarray): An array of bond maturities
        - coupons (np.ndaray): An array of bond coupons (Coupns are assumed to remain constant with respect to future payoffs)
        - prices (np.ndarray): An array of current market prices of bonds
        - coupon_dt (np.ndarray): An array of differences between times of coupon payments (e.g., for semi-annual coupon coupon_dt=0.5)
    ### Output:
        - Spot rates (np.ndarray) of the bonds provided
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Bootstrapping_(finance)
        - Original Source: N/A
    """
    # Arguments validation
    compare_array_len(array_1=principals, array_2=maturities, array_1_name="principals", array_2_name="maturities")
    compare_array_len(array_1=principals, array_2=coupons, array_1_name="principals", array_2_name="coupons")
    compare_array_len(array_1=principals, array_2=prices, array_1_name="principals", array_2_name="prices")
    compare_array_len(array_1=principals, array_2=coupon_dt, array_1_name="principals", array_2_name="coupon_dt")
    if sum(coupon_dt>1)+sum(coupon_dt<0)!=0:
        raise ValueError("The argument coupon_dt must have entries defined within the [0, 1] range.")
    # Bootstrap method
    spot_rates = np.array([])
    for i in range(len(principals)):
        payment_times_before_maturity = np.arange(start=coupon_dt[i], stop=maturities[i], step=coupon_dt[i])
        discount_term = 0
        for time_step in payment_times_before_maturity:
            discount_term = discount_term + np.exp(-time_step*spot_rates[np.nonzero(maturities==time_step)][0])
        spot_rate = -np.log((prices[i] - coupons[i]*coupon_dt[i]*discount_term)/(principals[i] + coupons[i]*coupon_dt[i]))/maturities[i]
        spot_rates = np.append(spot_rates, spot_rate)
    return spot_rates



class YtMMethod(Enum):
    NUMERICAL = 1
    APPROXIMATION = 2



class Bond(FinancialInstrumentInterface, BondInterface):
    """
    ## Description
    Bond financial instrument and its methods.
    ### Input:
        - bond_struct (BondStruct): Parameters for defining a Bond instance
        - financial_instrument_struct (FinancialInstrumentStruct): Parameters for defining regulatory categorization of an instrument
        - portfolio_instrument_struct (PortfolioInstrumentStruct): Parameters for defining historical data for portfolio construction and applications
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Bond_(finance)
        - Original Source: N/A
    """
    def __init__(self, bond_struct: BondStruct,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.CASH_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.DEBT_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]),
                                                                                                  portfolio_predicatble_income=np.array([]))) -> None:
        # Arguments validation
        type_check(value=bond_struct, type_=BondStruct, value_name="bond_struct")
        type_check(value=financial_instrument_struct, type_=FinancialInstrumentStruct, value_name="financial_instrument_stuct")
        type_check(value=portfolio_instrument_struct, type_=PortfolioInstrumentStruct, value_name="portfolio_instrument_struct")
        # BondStruct parameters
        self.bond_type = bond_struct.bond_type
        self.principal = bond_struct.principal
        self.coupon_rate = bond_struct.coupon_rate
        self.maturity = bond_struct.maturity
        self.initial_price = bond_struct.initial_price
        self.compounding_type = bond_struct.compounding_type
        self.compounding_frequency = bond_struct.compounding_frequency
        self.coupon_growth_rate = bond_struct.coupon_growth_rate
        self.inflation_rate = bond_struct.inflation_rate
        self.first_coupon_time = bond_struct.first_coupon_time
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_predictable_income = portfolio_instrument_struct.portfolio_predicatble_income
        # Derived parameters
        self.time_step = float(1/self.compounding_frequency)
        self.coupon = self.principal*self.coupon_rate/self.compounding_frequency
        cashflow = Cashflow(cashflow=self.coupon, final_time=self.maturity, start_time=self.first_coupon_time, time_step=self.time_step,
                            time_array=None, cashflow_growth_rate=self.coupon_growth_rate, inflation_rate=self.inflation_rate)
        self.cashflow = cashflow.cashflow
        self.time_array = cashflow.time_array
        if isinstance(bond_struct.discount_rate, float):
            self.discount_rate = float(bond_struct.discount_rate)*np.ones(len(self.cashflow))
        else:
            if (len(bond_struct.discount_rate)==len(self.cashflow)):
                self.dicount_rate = bond_struct.discount_rate
            else:
                raise ValueError("For the argument discount_rate of type np.ndarray, its length must be {} based on the given time parameters provided".format(len(self.cashflow)))
    
    def __str__(self):
        return f"Bond: {self.identifier}"

    def present_value(self) -> float:
        """
        ## Description
        Present values of the bond.
        ### Output:
            - Present value (float) of the bond's coupons and principal
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Bond_valuation
            - Original Source: N/A
        """
        present_value = 0
        # Present value of coupon payments
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.discount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            present_value = present_value + self.cashflow[i]*discount_term.compounding_term(time=self.time_array[i])
        # Present value of principal and coupon payments
        present_value = present_value + self.principal*Compounding(rate=self.discount_rate[-1], compounding_type=self.compounding_type,
                                                                   compounding_frequency=self.compounding_frequency).compounding_term(time=self.maturity)
        return present_value
    
    def net_present_value(self) -> float:
        """
        ## Description
        Net present value of the bond.
        ### Output:
            - Present value of the bond minus the initial price it took to purchase the bond (float)
        """
        return -self.initial_price + self.present_value()
    
    def future_value(self) -> float:
        """
        ## Description
        Future value of the bond.
        ### Output:
            - Future value (float) of the bond at it maturity (Computed from the present value of the bond)
        """
        future_multiplicator = 1
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.discount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            future_multiplicator = future_multiplicator/discount_term.compounding_term(time=self.time_step)
        return self.present_value()*future_multiplicator
    
    def yield_to_maturity(self, ytm_method: YtMMethod=YtMMethod.NUMERICAL) -> float:
        """
        ## Description
        Estimated total rate of return of the bond from current time to its maturity.
        ### Input:
            - ytm_method (YtMMethod): Method for evaluating yield-to-maturity (i.e., NUMERIACAL - for numerical solution of the IRR, APPROXIMATION - for approximation formula)
        ### Output:
            - Yield-to-Maturity (YtM) (float)
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Yield_to_maturity
            - Original Source: N/A
        """
        type_check(value=ytm_method, type_=YtMMethod, value_name="ytm_method")
        match ytm_method:
            case YtMMethod.APPROXIMATION:
                return (self.coupon + (self.principal-self.initial_price)/self.maturity)/(0.5*(self.principal+self.initial_price))
            case YtMMethod.NUMERICAL:
                cashflow = copy.deepcopy(self.cashflow)
                cashflow[-1] = cashflow[-1] + self.principal
                return internal_rate_of_return(initial_cashflow=self.initial_price, cashflow=cashflow, time_array=self.time_array,
                                               compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
    
    def yield_to_call(self, call_price: float) -> float:
        """
        ## Description
        Estimated total rate of return of a callable bond until it is called.
        ### Input:
            - call_price (float): Strike price of the callable bond
        ### Output:
            - Yield-to-Call (float)
        """
        call_price = float(call_price)
        cashflow = copy.deepcopy(self.cashflow)
        cashflow[-1] = cashflow[-1] + call_price
        return internal_rate_of_return(initial_cashflow=0.0, cashflow=self.cashflow, time_array=self.time_array,
                                       compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)

    def spot_rate(self, spot_rates: Union[np.ndarray, None]=None) -> float:
        """
        ## Description
        Spot rate of a bond computed based on cashflows of the bond discounted with the provided spot rates.
        ### Input:
            - spot_rates (Union[np.ndarray, None]): Spot rates for bonds with smaller maturities (Must be the same length as the number of coupons of the bond)
        ### Output:
            - Spot rate (float) of the bond
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Spot_contract#Bonds_and_swaps
            - Original Source: N/A
        """
        if self.bond_type == BondType.ZERO_COUPON_BOND:
            return -np.log(self.initial_price/self.principal)/self.maturity
        else:
            if isinstance(spot_rates, np.ndarray) is False:
                raise TypeError("The argument spot_rates must be defined as a np.ndarray type for non-zero-coupon bonds.")
            compare_array_len(array_1=self.cashflow[:-1], array_2=spot_rates, array_1_name="cashflow[:-1]", array_2_name="spot_rates")
            # Boostrap method based of provided spot_rates
            discounted_coupons = 0
            for i in range(len(self.time_array[:-1])):
                discount_term = Compounding(rate=spot_rates[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
                discounted_coupons = discounted_coupons + self.cashflow[i]*discount_term.compounding_term(time=self.time_array[i])
            match self.compounding_type:
                case CompoundingType.CONTINUOUS:
                    return -np.log((self.initial_price - discounted_coupons)/(self.principal + self.cashflow[-1]))/self.maturity
                case CompoundingType.PERIODIC:
                    return self.compounding_frequency*((self.initial_price - discounted_coupons)/(self.principal + self.cashflow[-1]))**(-1/(self.maturity*self.compounding_frequency)) - self.compounding_frequency
        
    def par_yield(self) -> float:
        """
        ## Description
        Estimated rate of return of the bond assuming that its market prics is equal to its principal.
        ### Output:
            - Par yield (float) of the bond
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Par_yield#
            - Original Source: N/A
        """
        discount_terms = 0
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.dicount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            discount_terms = discount_terms + discount_term.compounding_term(time=self.time_array[i])
        return self.compounding_frequency*(self.principal*(1-discount_term))/discount_terms

    def duration(self, modified: bool=False) -> float:
        """
        ## Description
        Weigthted average of the times when bond's payoffs are made.\n
        Note: This method computes Macaulay duration.
        ### Input:
            - modified (bool): Modifying the duration for periodic compounding
        ### Output:
            - Duration (float) of the bond
        ### LaTeX Formula:
            - B = \\sum^{n}_{i=1}c_{i}e^{-yt_{i}}
        ### Links:
            - Wikipedia:https://en.wikipedia.org/wiki/Bond_duration
            - Original Source: N/A
        """
        ytm = self.yield_to_maturity()
        weighted_cashflows = 0
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=ytm, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            cashflow = self.cashflow[i]
            # Principal included in calculation for final time step
            if i==(len(self.time_array)-1):
                cashflow = cashflow + self.principal
            weighted_cashflows = weighted_cashflows + self.time_array[i]*cashflow*discount_term.compounding_term(time=self.time_array[i])
        duration = weighted_cashflows/self.initial_price
        # Duration modified for periodic compounding
        if modified and (self.compounding_type==CompoundingType.PERIODIC):
            duration = duration/(1+ytm/self.compounding_frequency)
        return duration

    def convexity(self, modified: bool=False) -> float:
        """
        ## Description
        Measure of the curvature in relationship between bond's prices and yields.
        ### Input:
            - modified (bool): Modifying the convexity for periodic compounding
        ### Output:
            - Convexity (float) of the bond
        ### LaTeX Formula:
            - C = \\frac{1}{B}\\sum^{n}_{i=1}c_{i}t^{2}_{i}e^{-yt_{i}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Bond_convexity
            - Original Source: N/A
        """
        ytm = self.yield_to_maturity()
        weighted_cashflows = 0
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=ytm, compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            cashflow = self.cashflow[i]
            # Principal included in calculation for final time step
            if i==(len(self.time_array)-1):
                cashflow = cashflow + self.principal
            time = self.time_array[i]**2
            # Convexity modified for periodic compounding
            if modified and (self.compounding_type==CompoundingType.PERIODIC):
                time = time + self.time_array[i]/self.compounding_frequency
            weighted_cashflows = weighted_cashflows + time*cashflow*discount_term.compounding_term(time=self.time_array[i])
        convexity = weighted_cashflows/self.initial_price
        if modified and (self.compounding_type==CompoundingType.PERIODIC):
            convexity = convexity/((1+ytm/self.compounding_frequency)**2)
        return convexity
    
    def accrued_interest(self, time_since_last_coupon: float, time_separation: float) -> float:
        """
        ## Description
        Amount of coupon accumulated before the purchase of the bond.
        ### Input:
            - time_since_last_coupon (float): Period of time passed from last coupon payment
            - time_separation (float): Period of time between consecutive bond payments
        ### Output:
            - Accrued interest (float) of the bond
        ### LaTeX Formula:
            - \\textit{Accrued Interest} = \\frac{Coupon}{\\textit{Compounding Frequency}}\\frac{\\textit{Time since last coupon payment}}{\\textit{Time separating coupon payments}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Accrued_interest
            - Original Source: N/A
        """
        return (self.coupon/self.compounding_frequency) * (float(time_since_last_coupon)/float(time_separation))

    def hazard_rate(self, yield_spread: float, recovery_rate: float=0.4) -> float:
        """
        ## Description
        Frequency with which the failure of bond repayment occurs.\n
        Note: It is assumed that the excess yield is the compensation for the possibility of default.
        ### Input:
            - yield_spread (float): Yield spread of the bond
            - recovery_rate (float): Assumed amount of bond's value that will be recovered
        ### Output:
            - Hazard rate (float) of the bond
        ### LaTeX Formula:
            - \\textit{Hazard Rate} = \\frac{\\textit{Yield Spread}}{1 - \\text{Recovery Rate}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Failure_rate
            - Original Source: N/A
        """
        return float(yield_spread) / (1 + float(recovery_rate))