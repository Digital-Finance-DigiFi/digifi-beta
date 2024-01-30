from enum import Enum
import numpy as np
import scipy
from src.digifi.utilities.general_utils import (compare_array_len, type_check)



class CAPMType(Enum):
    STANDARD = 1
    THREE_FACTOR_FAMA_FRENCH = 2
    FIVE_FACTOR_FAMA_FRENCH = 3



class CAPMSolutionType(Enum):
    LINEAR_REGRESSION = 1
    COVARIANCE = 2



class CAPM:
    """
    CAPM, three-factor and five-factor Famma-French models.
    Contains methods for finding asset beta and predicting expected asset returns with the given beta.
    """
    def __init__(self, capm_type: CAPMType, solution_type: CAPMSolutionType=CAPMSolutionType.LINEAR_REGRESSION) -> None:
        # Arguments validation
        type_check(value=capm_type, type_=CAPMType, value_name="capm_type")
        type_check(value=solution_type, type_=CAPMSolutionType, value_name="solution_type")
        if (capm_type!=CAPMType.STANDARD) and (solution_type==CAPMSolutionType.COVARIANCE):
            raise ValueError("The covariance solution method is only available for the standard CAPM")
        self.capm_type = capm_type
        self.solution_type = solution_type
    
    def linear_regression(self, market_returns: np.ndarray, rf_returns: np.ndarray, alpha: float, beta: float,
                          smb: np.ndarray=np.array([]), s_beta: float=0.0, hml: np.ndarray=np.array([]), h_beta: float=0.0,
                          rmw: np.ndarray=np.array([]), r_beta: float=0.0, cma: np.ndarray=np.array([]), c_beta: float=0.0) -> np.ndarray:
        """
        E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA.
        Computes the expected return of an asset/project given the risk-free rate, expected market return, SMB, HML, RMW, CMA and their betas.
            - SMB: difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
            - HML: difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
            - RMW: difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
            - CMA: difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
        """
        compare_array_len(array_1=market_returns, array_2=rf_returns, array_1_name="market_returns", array_2_name="rf_returns")
        lin_reg = float(alpha) + float(beta)*(market_returns - rf_returns)
        if self.capm_type!=CAPMType.STANDARD:
            compare_array_len(array_1=market_returns, array_2=smb, array_1_name="market_returns", array_2_name="smb")
            compare_array_len(array_1=market_returns, array_2=hml, array_1_name="market_returns", array_2_name="hml")
            lin_reg = lin_reg + float(s_beta)*smb + float(h_beta)*hml
        if self.capm_type==CAPMType.FIVE_FACTOR_FAMA_FRENCH:
            compare_array_len(array_1=market_returns, array_2=rmw, array_1_name="market_returns", array_2_name="rmw")
            compare_array_len(array_1=market_returns, array_2=cma, array_1_name="market_returns", array_2_name="cma")
            lin_reg = lin_reg + float(r_beta)*rmw + float(c_beta)*cma
        return lin_reg
    
    def linear_regression_train(self, asset_returns: np.ndarray, market_returns: np.ndarray, rf_returns: np.ndarray,
                          smb: np.ndarray=np.array([]), hml: np.ndarray=np.array([]),
                          rmw: np.ndarray=np.array([]), cma: np.ndarray=np.array([])) -> dict:
        """
        Finds the values of parameters alpha and betas (if COVARIANCE solution type is used, only beta is returned).
        """
        compare_array_len(array_1=asset_returns, array_2=market_returns, array_1_name="asset_returns", array_2_name="market_returns")
        if self.solution_type==CAPMSolutionType.COVARIANCE:
            cov_matrix = np.cov(asset_returns, market_returns, ddof=0)
            return {"beta":float(cov_matrix[1, 0]/cov_matrix[1,1])}
        else:
            y = scipy.array(asset_returns)
            match self.capm_type:
                case CAPMType.STANDARD:
                    def line(x, alpha, beta):
                        return self.linear_regression(market_returns=x[0], rf_returns=x[1], alpha=alpha, beta=beta)
                    x = scipy.array([market_returns, rf_returns])
                    popt, _ = scipy.optimize.curve_fit(line, x, y)
                    return {"alpha":popt[0], "beta":popt[1]}
                case CAPMType.THREE_FACTOR_FAMA_FRENCH:
                    def line(x, alpha, beta, s_beta, h_beta):
                        return self.linear_regression(market_returns=x[0], rf_returns=x[1], alpha=alpha, beta=beta,
                                                      smb=x[2], s_beta=s_beta, hml=x[3], h_beta=h_beta)
                    x = scipy.array([market_returns, rf_returns, smb, hml])
                    popt, _ = scipy.optimize.curve_fit(line, x, y)
                    return {"alpha":popt[0], "beta":popt[1], "s_beta":popt[2], "h_beta":popt[3]}
                case CAPMType.FIVE_FACTOR_FAMA_FRENCH:
                    def line(x, alpha, beta, s_beta, h_beta, r_beta, c_beta):
                        return self.linear_regression(market_returns=x[0], rf_returns=x[1], alpha=alpha, beta=beta,
                                                      smb=x[2], s_beta=s_beta, hml=x[3], h_beta=h_beta,
                                                      rmw=x[4], r_beta=r_beta, cma=x[5], c_beta=c_beta)
                    x = scipy.array([market_returns, rf_returns, smb, hml, rmw, cma])
                    popt, _ = scipy.optimize.curve_fit(line, x, y)
                    return {"alpha":popt[0], "beta":popt[1], "s_beta":popt[2], "h_beta":popt[3], "r_beta":popt[4], "c_beta":popt[5]}
            