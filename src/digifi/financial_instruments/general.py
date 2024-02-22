import abc
from dataclasses import dataclass
from enum import Enum
from digifi.utilities.general_utils import DataClassValidation



class FinancialInstrumentType(Enum):
    # Values are determined by the markets: stocks, bonds, deposits, loans, checks
    CASH_INSTRUMENT = 1
    # Options, futures, forwards
    DERIVATIVE_INSTRUMENT = 2



class FinancialInstrumentAssetClass(Enum):
    # Loans, certificates of deposit, bank deposits, futures, forwards, options, bonds, mortgage-backed securities
    DEBT_BASED_INSTRUMENT = 1
    # Stocks - common and preferred, some ETFs, mutual funds, REITs, checks, limited partnerships
    EQUITY_BASED_INSTRUMENT = 2
    # Forwards, futures, options, CFDs, swaps
    FOREIGN_EXCHANGE_INSTRUMENT = 3



@dataclass(slots=True)
class FinancialInstrumentStruct(DataClassValidation):
    """
    ## Description
    Struct with general financial instrument data.
    ### Input:
        - instrument_type (FinancialInstrumentType): Type of the financial instrument (i.e., cash instrument or derivative instrument)
        - asset_class (FinancialInstrumentAssetClass): Asset class of the instrument (i.e., debt-based, equity-based or foreign exchange instrument)
        - identifier (str): Financial Instrument Global Identifier (FIGI), International Securities Identification Number (ISIN)
    """
    instrument_type: FinancialInstrumentType
    asset_class: FinancialInstrumentAssetClass
    identifier: str



class FinancialInstrumentInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "present_value") and
                callable(subclass.present_value) and
                hasattr(subclass, "net_present_value") and
                callable(subclass.net_present_value) and
                hasattr(subclass, "future_value") and
                callable(subclass.future_value))
    
    @abc.abstractmethod
    def present_value(self) -> float:
        """
        ## Description
        Calculate present value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def net_present_value(self) -> float:
        """
        ## Description
        Calculate net present value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def future_value(self) -> float:
        """
        ## Description
        Calculate future value.
        """
        raise NotImplementedError