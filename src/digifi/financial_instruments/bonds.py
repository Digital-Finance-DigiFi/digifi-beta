from typing import Union
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
from src.digifi.utilities.time_value_utils import Cashflow, CompoundingType, Compounding
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
    discount_rate: float
    maturity: float
    initial_investment: float



class BondInteraface(metaclass=abc.ABCMeta):
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
    def zero_rate(self) -> float:
        """
        Calculate zero rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def par_yield(self) -> float:
        """
        Calculate par yield.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def duration(self, modified:bool=False) -> float:
        """
        Calculate duration.
        """
        raise NotImplementedError
    
    # TODO: Potentially add modified duration method
    
    @abc.abstractmethod
    def convexity(self) -> float:
        """
        Calculate convexity.
        """
        raise NotImplementedError



class Bond(FinancialInstrumentInterface, BondStruct):#, BondInteraface):
    """
    Bond financial instrument and its methods.
    """
    def __init__(self, bond_type: BondType, principle: float, coupon_rate: float, discount_rate: Union[np.ndarray, float], maturity: float,
                 initial_investment:float, compounding_type: CompoundingType=CompoundingType.PERIODIC, compounding_frequency: int=1,
                 identifier: Union[str, int]="0", first_coupon_time: float=1.0, time_step: float=1.0, coupon_growth_rate: float=0.0,
                 inflation_rate: float=0.0) -> None:
        # Bond class parameters
        self.bond_type = bond_type
        self.compounding_type = compounding_type
        self.coupon_growth_rate = float(coupon_growth_rate)
        self.compounding_frequency = int(compounding_frequency)
        self.inflation_rate = float(inflation_rate)
        self.first_coupon_time = float(first_coupon_time)
        self.time_step = float(time_step)
        # BondStruct parameters
        self.principal = float(principle)
        self.coupon_rate = float(coupon_rate)
        self.maturity = float(maturity)
        initial_investment = float(initial_investment)
        # FinancialInstrumentStruct parameters
        self.instrument_type = FinancialInstrumentType.CASH_INSTRUMENT
        self.asset_class = FinancialInstrumentAssetClass.DEBT_BASED_INSTRUMENT
        self.identifier = identifier
        # Derived parameters
        self.coupon = principle*coupon_rate
        cashflow = Cashflow(cashflow=self.coupon, final_time=maturity, start_time=first_coupon_time, time_step=time_step, time_array=None,
                            cashflow_growth_rate=coupon_growth_rate, inflation_rate=inflation_rate)
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
        return -self.initial_investment + self.present_value()
    
    def future_value(self) -> float:
        future_multiplicator = 1
        for i in range(len(self.time_array)):
            discount_term = Compounding(rate=self.discount_rate[i], compounding_type=self.compounding_type, compounding_frequency=self.compounding_frequency)
            future_multiplicator = future_multiplicator/discount_term.compounding_term(time=1)
        return self.present_value()*future_multiplicator
    
    # TODO: Implement bond interface