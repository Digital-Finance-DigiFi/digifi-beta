from typing import Union
import abc
from dataclasses import dataclass
import numpy as np
from src.digifi.utilities.time_value_utils import Cashflow, CompoundingType, Compounding
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from src.digifi.portfolio_applications.general import PortfolioInstrumentStruct



@dataclass
class BillOfExchangeStruct(FinancialInstrumentStruct, PortfolioInstrumentStruct):
    principal: float
    coupon_rate: float
    discount_rate: float
    maturity: float
    initial_investment: float