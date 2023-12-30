from typing import Union
import abc
from dataclasses import dataclass
import numpy as np
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)



@dataclass
class StockStruct(FinancialInstrumentStruct):
    # TODO: Add constraint for only historical prices to be allowed to be accessed
    price_per_share: Union[np.ndarray, float]
    dividend: Union[np.ndarray, float]
    earnings_per_share: Union[np.ndarray, float]



class StockInteraface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "cost_of_equity_capital") and
                callable(subclass.cost_of_equity_capital) and
                hasattr(subclass, "dividend_discount_model") and
                callable(subclass.dividend_discount_model) and
                hasattr(subclass, "payout_ratio") and
                callable(subclass.payout_ratio) and
                hasattr(subclass, "plowback_ratio") and
                callable(subclass.plowback_ratio) and
                hasattr(subclass, "return_on_equity") and
                callable(subclass.return_on_equity) and
                hasattr(subclass, "dividend_growth_rate") and
                callable(subclass.dividend_growth_rate) and
                hasattr(subclass, "present_value_of_growth_opportunities") and
                callable(subclass.present_value_of_growth_opportunities) and
                hasattr(subclass, "valuation") and
                callable(subclass.valuation))
    
    @abc.abstractmethod
    def cost_of_equity_capital(self) -> float:
        """
        Calculate cost of equity capital.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def dividend_discount_model(self) -> float:
        """
        Create dividend discount model.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def payout_ratio(self) -> float:
        """
        Calculate payout ratio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def plowback_ratio(self, modified:bool=False) -> float:
        """
        Calculate plowback ratio.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def return_on_equity(self) -> float:
        """
        Calculate return on equity.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def dividend_growth_rate(self) -> float:
        """
        Calculate dividend growth rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def present_value_of_growth_opportunities(self) -> float:
        """
        Calculate present value of growth opportunities.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def valuation(self) -> float:
        """
        Valuate stock.
        """
        # Valuation by comparables - P/E, Market-to-Book Ratio, etc.; discounted cash flow - post-horizon PVGO
        raise NotImplementedError
    


class Stock(FinancialInstrumentInterface, StockStruct, StockInteraface):
    """
    Stock financial instrument and its methods.
    """
    def __init__(self) -> None:
        return None
    
    # TODO: Implement financial instrument and stock interfaces