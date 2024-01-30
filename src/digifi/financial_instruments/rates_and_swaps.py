from dataclasses import dataclass
from src.digifi.utilities.general_utils import DataClassValidation
from src.digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from src.digifi.portfolio_applications.general import PortfolioInstrumentStruct



@dataclass(slots=True)
class ForwardRateAgreementStruct(DataClassValidation):
    fixed_rate: float
    forward_rate: float
    time_to_maturity: float
    principal: float

    def validate_time_to_maturity(self, value: float, **_) -> float:
        if value<0:
            raise ValueError("The parameter time_to_maturity must be positive.")
        return value


class ForwardRateAgreement(FinancialInstrumentInterface):
    """
    Forward rate agreement and its methods.
    """
    def __init__(self) -> None:
        return None
    
    # TODO: Implement financial instrument and forward rate agreement interfaces