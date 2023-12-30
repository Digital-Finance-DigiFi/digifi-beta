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
    
    def capm_get_params(self, asset_returns: np.ndarray, market_returns: np.ndarray, rf_rates: np.ndarray,
                      solution_type: CAPMSolutionType=CAPMSolutionType.LINEAR_REGRESSION) -> dict:
        """
        E[R_{A}] = \\alpha + R_{rf} + \\beta(E[R_{M}] - R_{rf}) + \\epsilon.
        Finds the values of parameters alpha and beta (if COVARIANCE solution type is used, only beta is returned).
        """
        compare_array_len(array_1=asset_returns, array_2=market_returns, array_1_name="asset_returns", array_2_name="market_returns")
        compare_array_len(array_1=asset_returns, array_2=rf_rates, array_1_name="asset_returns", array_2_name="rf_rates")
        match solution_type:
            case CAPMSolutionType.LINEAR_REGRESSION:
                # TODO: Add CAPM linear regression model
                return {"alpha":np.nan, "beta":np.nan}
            case CAPMSolutionType.COVARIANCE:
                cov_matrix = np.cov(asset_returns, market_returns, ddof=0)
                return {"beta":float(cov_matrix[1, 0]/cov_matrix[1,1])}
        raise ValueError("The argument solution_type must be of CAPMSolutionType type.")
                
    
    @staticmethod
    def capm_get_asset_return(market_returns: Union[np.ndarray, float], rf_rates: Union[np.ndarray, float],
                              betas: Union[np.ndarray, float], alphas: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta(E[R_{M}] - R_{rf}).
        Computes the expected return of an asset/project given the risk-free rate, expected market return and beta.
        """
        if (isinstance(market_returns, np.ndarray) and isinstance(rf_rates, np.ndarray) and isinstance(betas, np.ndarray) and
            isinstance(alphas, np.ndarray)):
            compare_array_len(array_1=rf_rates, array_2=market_returns, array_1_name="rf_rates", array_2_name="market_returns")
            compare_array_len(array_1=rf_rates, array_2=betas, array_1_name="rf_rates", array_2_name="betas")
            compare_array_len(array_1=rf_rates, array_2=alphas, array_1_name="rf_rates", array_2_name="alphas")
        elif isinstance(market_returns, float) and isinstance(rf_rates, float) and isinstance(betas, float) and isinstance(alphas, float):
            pass
        else:
            raise ValueError("The arguments market_returns, rf_rates and betas all have to be simultaneously either an np.ndarray or float.")
        return rf_rates + alphas + betas*(market_returns-rf_rates)
    
    def tfff_get_params(self,) -> float:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\epsilon.
        Finds the values of parameters alpha and betas.
            - SMB: difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
            - HML: difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
        """
        # TODO: Add three-factor Famma-French implementation

    @staticmethod
    def tfff_get_aset_return(market_returns: Union[np.ndarray, float], rf_rates: Union[np.ndarray, float], m_betas: Union[np.ndarray, float],
                             smb: Union[np.ndarray, float], s_betas: Union[np.ndarray, float],
                             hml: Union[np.ndarray, float], h_betas: Union[np.ndarray, float],
                             alphas: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML.
        Computes the expected return of an asset/project given the risk-free rate, expected market return, SMB, HML and their betas.
            - SMB: difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
            - HML: difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
        """
        if (isinstance(market_returns, np.ndarray) and isinstance(rf_rates, np.ndarray) and isinstance(m_betas, np.ndarray) and
            isinstance(smb, np.ndarray) and isinstance(s_betas, np.ndarray) and isinstance(hml, np.ndarray) and isinstance(h_betas, np.ndarray) and
            isinstance(alphas, np.ndarray)):
            compare_array_len(array_1=rf_rates, array_2=market_returns, array_1_name="rf_rates", array_2_name="market_returns")
            compare_array_len(array_1=rf_rates, array_2=m_betas, array_1_name="rf_rates", array_2_name="m_betas")
            compare_array_len(array_1=rf_rates, array_2=smb, array_1_name="rf_rates", array_2_name="smb")
            compare_array_len(array_1=rf_rates, array_2=s_betas, array_1_name="rf_rates", array_2_name="s_betas")
            compare_array_len(array_1=rf_rates, array_2=hml, array_1_name="rf_rates", array_2_name="hml")
            compare_array_len(array_1=rf_rates, array_2=h_betas, array_1_name="rf_rates", array_2_name="h_betas")
            compare_array_len(array_1=rf_rates, array_2=alphas, array_1_name="rf_rates", array_2_name="alphas")
        elif (isinstance(market_returns, float) and isinstance(rf_rates, float) and isinstance(m_betas, float) and
              isinstance(smb, float) and isinstance(s_betas, float) and isinstance(hml, float) and isinstance(h_betas, float) and
              isinstance(alphas, float)):
            pass
        else:
            raise ValueError("The arguments market_returns, rf_rates and betas all have to be simultaneously either an np.ndarray or float.")
        return rf_rates + alphas + m_betas*(market_returns-rf_rates) + s_betas*smb + h_betas*hml

    def ffff_get_params(self,) -> float:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA + \\epsilon.
        Finds the values of parameters alpha and betas.
            - SMB: difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
            - HML: difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
            - RMW: difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
            - CMA: difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
        """
        # TODO: Add five-factor Famma-French implementation

    @staticmethod
    def ffff_get_asset_return(market_returns: Union[np.ndarray, float], rf_rates: Union[np.ndarray, float], m_betas: Union[np.ndarray, float],
                              smb: Union[np.ndarray, float], s_betas: Union[np.ndarray, float],
                              hml: Union[np.ndarray, float], h_betas: Union[np.ndarray, float],
                              rmw: Union[np.ndarray, float], r_betas: Union[np.ndarray, float],
                              cma: Union[np.ndarray, float], c_betas: Union[np.ndarray, float],
                              alphas: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA.
        Computes the expected return of an asset/project given the risk-free rate, expected market return, SMB, HML, RMW, CMA and their betas.
            - SMB: difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
            - HML: difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
            - RMW: difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
            - CMA: difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
        """
        if (isinstance(market_returns, np.ndarray) and isinstance(rf_rates, np.ndarray) and isinstance(m_betas, np.ndarray) and
            isinstance(smb, np.ndarray) and isinstance(s_betas, np.ndarray) and isinstance(hml, np.ndarray) and isinstance(h_betas, np.ndarray) and
            isinstance(rmw, np.ndarray) and isinstance(r_betas, np.ndarray) and isinstance(cma, np.ndarray) and isinstance(c_betas, np.ndarray) and
            isinstance(alphas, np.ndarray)):
            compare_array_len(array_1=rf_rates, array_2=market_returns, array_1_name="rf_rates", array_2_name="market_returns")
            compare_array_len(array_1=rf_rates, array_2=m_betas, array_1_name="rf_rates", array_2_name="m_betas")
            compare_array_len(array_1=rf_rates, array_2=smb, array_1_name="rf_rates", array_2_name="smb")
            compare_array_len(array_1=rf_rates, array_2=s_betas, array_1_name="rf_rates", array_2_name="s_betas")
            compare_array_len(array_1=rf_rates, array_2=hml, array_1_name="rf_rates", array_2_name="hml")
            compare_array_len(array_1=rf_rates, array_2=h_betas, array_1_name="rf_rates", array_2_name="h_betas")
            compare_array_len(array_1=rf_rates, array_2=rmw, array_1_name="rf_rates", array_2_name="rmw")
            compare_array_len(array_1=rf_rates, array_2=r_betas, array_1_name="rf_rates", array_2_name="r_betas")
            compare_array_len(array_1=rf_rates, array_2=cma, array_1_name="rf_rates", array_2_name="cma")
            compare_array_len(array_1=rf_rates, array_2=c_betas, array_1_name="rf_rates", array_2_name="c_betas")
            compare_array_len(array_1=rf_rates, array_2=alphas, array_1_name="rf_rates", array_2_name="alphas")
        elif (isinstance(market_returns, float) and isinstance(rf_rates, float) and isinstance(m_betas, float) and
              isinstance(smb, float) and isinstance(s_betas, float) and isinstance(hml, float) and isinstance(h_betas, float) and
              isinstance(rmw, float) and isinstance(r_betas, float) and isinstance(cma, float) and isinstance(c_betas, float) and
              isinstance(alphas, float)):
            pass
        else:
            raise ValueError("The arguments market_returns, rf_rates and betas all have to be simultaneously either an np.ndarray or float.")
        return rf_rates + alphas + m_betas*(market_returns-rf_rates) + s_betas*smb + h_betas*hml + r_betas*rmw + c_betas*cma
