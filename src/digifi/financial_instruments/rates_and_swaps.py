from dataclasses import dataclass
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)



@dataclass
class ForwardRateAgreementStruct(FinancialInstrumentStruct):
    fixed_rate: float
    forward_rate: float
    term: float
    principal: float



class ForwardRateAgreement(FinancialInstrumentInterface, ForwardRateAgreementStruct):
    """
    Forward rate agreement and its methods.
    """
    def __init__(self) -> None:
        return None
    
    # TODO: Implement financial instrument and forward rate agreement interfaces