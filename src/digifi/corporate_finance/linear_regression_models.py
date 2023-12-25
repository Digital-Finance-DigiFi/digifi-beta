from typing import Union
from enum import Enum
import numpy as np
from src.digifi.utilities.general_utils import compare_array_len



class CAPMSolutionType(Enum):
        LINEAR_REGRESSION = 1
        COVARIANCE = 2



class CAPM:
    """
    CAPM, three-factor and five-factor Famma-French models.
    Contains methods for finding asset beta and predicting expected asset returns with the given beta.
    """
    def __init__(self) -> None:
        # CAPM arguments
        self.asset_returns = np.array([])
        self.market_returns = np.array([])
        self.rf_rates = np.array([])
        self.beta = np.nan
        # Three-factor Famma-French arguments
        # TODO: Add three-factor Famma-French arguments
        self.smb = np.array([])
        self.hml = np.array([])
        # Five-factor Famma-French arguments
        # TODO: Add five factor Famma-French arfuments
    
    def capm_get_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray, rf_rates: np.ndarray,
                      solution_type: CAPMSolutionType=CAPMSolutionType.LINEAR_REGRESSION) -> float:
        compare_array_len(array_1=asset_returns, array_2=market_returns, array_1_name="asset_returns", array_2_name="market_returns")
        compare_array_len(array_1=asset_returns, array_2=rf_rates, array_1_name="asset_returns", array_2_name="rf_rates")
        # TODO: Add CAPM linear regression model
        match solution_type:
            case CAPMSolutionType.LINEAR_REGRESSION:
                # TODO: Add CAPM linear regression model
                pass
            case CAPMSolutionType.COVARIANCE:
                cov_matrix = np.cov(asset_returns, market_returns, ddof=0)
                return float(cov_matrix[1, 0]/cov_matrix[1,1])
        raise ValueError("The argument solution_type must be of CAPMSolutionType type.")
                
    
    @staticmethod
    def capm_get_asset_return(market_returns: Union[np.ndarray, float], rf_rates: Union[np.ndarray, float],
                              betas: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if isinstance(market_returns, np.ndarray) and isinstance(rf_rates, np.ndarray) and isinstance(betas, np.ndarray):
            compare_array_len(array_1=rf_rates, array_2=market_returns, array_1_name="rf_rates", array_2_name="market_returns")
            compare_array_len(array_1=rf_rates, array_2=betas, array_1_name="rf_rates", array_2_name="betas")
        elif isinstance(market_returns, float) and isinstance(rf_rates, float) and isinstance(betas, float):
            pass
        else:
            raise ValueError("The arguments market_returns, rf_rates and betas all have to be simultaneously either an np.ndarray or float.")
        return rf_rates + betas*(market_returns-rf_rates)
