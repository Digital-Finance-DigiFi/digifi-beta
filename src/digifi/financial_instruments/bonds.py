from typing import Union
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import compare_array_len
from src.digifi.utilities.time_value_utils import (Cashflow, CompoundingType, Compounding, internal_rate_of_return)
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)



class BondType(Enum):
    ANNUITY_BOND = 1
    GROWING_ANNUITY_BOND = 2
    ZERO_COUPON_BOND = 3
    CONVERTIBLE_BOND = 4
    CALLABLE_BOND = 5
    PUTTABLE_BOND = 6



@dataclass
class BondStruct(FinancialInstrumentStruct):
    principal: float
    coupon_rate: float
    discount_rate: Union[np.ndarray, float]
    maturity: float
    initial_price: float



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
                callable(subclass.convexity))
    
    @abc.abstractmethod
    def yield_to_maturity(self) -> float:
        """
        Calculate yield to maturity.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def spot_rate(self) -> float:
        """
        Calculate spot rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def par_yield(self) -> float:
        """
        Calculate par yield.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def duration(self) -> float:
        """
        Calculate duration.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def convexity(self) -> float:
        """
        Calculate convexity.
        """
        raise NotImplementedError



class YtMMethod(Enum):
    NUMERICAL = 1
    APPROXIMATION = 2



class Bond(FinancialInstrumentInterface, BondStruct, BondInterface):
    """
    Bond financial instrument and its methods.
    """
    def __init__(self, bond_type: BondType, principal: float, coupon_rate: float, discount_rate: Union[np.ndarray, float], maturity: float,
                 initial_price: float=0.0, compounding_type: CompoundingType=CompoundingType.PERIODIC, compounding_frequency: int=1,
                 identifier: Union[str, int]="0", first_coupon_time: float=1.0, coupon_growth_rate: float=0.0,
                 inflation_rate: float=0.0) -> None:
        # TODO: Add convertible, callable and puttable bonds
        # Bond class parameters
        self.bond_type = bond_type
        self.compounding_type = compounding_type
        self.coupon_growth_rate = float(coupon_growth_rate)
        self.compounding_frequency = int(compounding_frequency)
        self.inflation_rate = float(inflation_rate)
        self.first_coupon_time = float(first_coupon_time)
        # BondStruct parameters
        self.principal = float(principal)
        self.coupon_rate = float(coupon_rate)
        self.maturity = float(maturity)
        self.initial_price = float(initial_price)
        # FinancialInstrumentStruct parameters
        self.instrument_type = FinancialInstrumentType.CASH_INSTRUMENT
        self.asset_class = FinancialInstrumentAssetClass.DEBT_BASED_INSTRUMENT
        self.identifier = identifier
        # Derived parameters
        self.time_step = float(1/compounding_frequency)
        self.coupon = self.principal*self.coupon_rate/self.compounding_frequency
        cashflow = Cashflow(cashflow=self.coupon, final_time=maturity, start_time=first_coupon_time, time_step=self.time_step,
                            time_array=None, cashflow_growth_rate=coupon_growth_rate, inflation_rate=inflation_rate)
        self.cashflow = cashflow.cashflow
        self.time_array = cashflow.time_array
        if isinstance(discount_rate, float):
            self.discount_rate = float(discount_rate)*np.ones(len(self.cashflow))
        elif isinstance(discount_rate, np.ndarray):
            if (len(discount_rate)==len(self.cashflow)):
                self.dicount_rate = discount_rate
            else:
                raise ValueError("For the argument discount_rate of type np.ndarray, its length must be {} based on the given time parameters provided".format(len(self.cashflow)))
        else:
            raise TypeError("The argument discount_rate must be either of type float or np.ndarray.")

    def present_value(self) -> float:
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
        return -self.initial_price + self.present_value()
    
    def future_value(self) -> float:
        future_multiplicator = 1
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.discount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            future_multiplicator = future_multiplicator/discount_term.compounding_term(time=self.time_step)
        return self.present_value()*future_multiplicator
    
    def yield_to_maturity(self, ytm_method: YtMMethod=YtMMethod.NUMERICAL) -> float:
        match ytm_method:
            case YtMMethod.APPROXIMATION:
                return (self.coupon + (self.principal-self.initial_price)/self.maturity)/(0.5*(self.principal+self.initial_price))
            case YtMMethod.NUMERICAL:
                return internal_rate_of_return(initial_cashflow=self.initial_price, cashflow=self.cashflow, time_array=self.time_array,
                                               compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
        raise ValueError("The argument ytm_method must be of YtMMethod type.")
    
    def spot_rate(self, spot_rates: Union[np.ndarray, None]=None) -> float:
        """
        Spot rate of a bond computed based on cashflows of the bond discounted with the provided spot rates.
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
            raise ValueError("The argument compounding_type must be of CompoundingType type.")
        
    def par_yield(self) -> float:
        discount_terms = 0
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.dicount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            discount_terms = discount_terms + discount_term.compounding_term(time=self.time_array[i])
        return self.compounding_frequency*(self.principal*(1-discount_term))/discount_terms

    def duration(self, modified: bool=False) -> float:
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