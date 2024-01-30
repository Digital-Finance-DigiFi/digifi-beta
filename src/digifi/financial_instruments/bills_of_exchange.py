from typing import Union
import abc
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.general_utils import DataClassValidation
from src.digifi.utilities.time_value_utils import Cashflow, CompoundingType, Compounding
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from src.digifi.portfolio_applications.general import PortfolioInstrumentStruct



@dataclass(slots=True)
class BillOfExchangeStruct(DataClassValidation):
    principal: float
    coupon_rate: float
    discount_rate: float
    maturity: float
    initial_investment: float

    def validate_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The maturity must be positive.")
        return value



class BillOfExchange(FinancialInstrumentInterface):
    """
    Bill of exchange financial instrument and its methods.
    """
    def __init__(self) -> None:
        pass
    # TODO: Implements BillOfEchange class