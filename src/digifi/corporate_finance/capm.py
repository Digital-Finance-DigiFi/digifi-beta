from typing import (Union, Any)
from enum import Enum
from dataclasses import dataclass
import numpy as np
import scipy
from digifi.utilities.general_utils import (compare_array_len, DataClassValidation)



class CAPMType(Enum):
    STANDARD = 1
    THREE_FACTOR_FAMA_FRENCH = 2
    FIVE_FACTOR_FAMA_FRENCH = 3



class CAPMSolutionType(Enum):
    LINEAR_REGRESSION = 1
    COVARIANCE = 2



@dataclass(slots=True)
class CAPMParams(DataClassValidation):
    """
    ## Description
    Parameters for the CAPM class.
    ### Input:
        - capm_type (CAPMType): Type of CAPM model (i.e., STANDARD, THREE_FACTOR_FAMA_FRENCH, ot FIVE_FACTOR_FAMA_FRENCH)
        - solution_type (CAPMSolutionType): Type of solution to use (i.e., COVARIANCE - works only for STANDARD CAPM model, or LINEAR_REGRESSION)
        - market_returns (np.ndarray): Returns of the market
        - rf (np.ndarray): Risk-free rate of return
        - SMB (Union[np.ndarray, None]): difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
        - HML (Union[np.ndarray, None]): difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
        - RMW (Union[np.ndarray, None]): difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
        - CMA (Union[np.ndarray, None]): difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
    """
    capm_type: CAPMType
    solution_type: CAPMSolutionType
    market_returns: np.ndarray
    rf: np.ndarray
    smb: Union[np.ndarray, None] = None
    hml: Union[np.ndarray, None] = None
    rmw: Union[np.ndarray, None] = None
    cma: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        super(CAPMParams, self).__post_init__()
        if (self.capm_type!=CAPMType.STANDARD) and (self.solution_type==CAPMSolutionType.COVARIANCE):
            raise ValueError("The covariance solution method is only available for the standard CAPM")
        compare_array_len(array_1=self.market_returns, array_2=self.rf, array_1_name="market_returns", array_2_name="rf")



def r_square(beta: float, sigma_market: float, sigma_epsilon: float) -> float:
    """
    ## Description
    The ratio of the systematic variance to the total variance.
    ### Input:
        - beta (float): Beta coefficient of the model
        - sigma_market (float): Standard deviation of market returns
        - sigma_epsilon (float): Standard deviation of the error in the model
    ### Output:
        - r_square (float): The ratio of exaplained variance to all variance
    ### LaTeX Formula:
        - R^{2} = \\frac{\\beta^{2}\\sigma^{2}_{M}}{\\beta^{2}\\sigma^{2}_{M} + \\sigma^{2}(\\epsilon)}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Coefficient_of_determination
        - Original Source: N/A
    """
    beta = float(beta)
    sigma_market = float(sigma_market)
    sigma_epsilon = float(sigma_epsilon)
    return (beta**2 * sigma_market**2) / ((beta**2 * sigma_market**2) + sigma_epsilon**2)

def adjusted_r_square(beta: float, sigma_market: float, sigma_epsilon: float, sample_size: int, k_variables: int) -> float:
    """
    ## Description
    Adjusted R-square for the upward bias in the R-square due to estimated values of the parameters used.
    ### Input:
        - beta (float): Beta coefficient of the model
        - sigma_market (float): Standard deviation of market returns
        - sigma_epsilon (float): Standard deviation of the error in the model
        - sample_size (int): Number of points used in the model
        - k_variables (int): Number of variables in the model
    ### Output:
        - adjusted_r_square (float): R-square adjusted for upward estimation bias
    ### LaTeX Formula:
        - R^{2}_{A} = 1 - (1-R^{2})\\frac{n-1}{n-k-1}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        - Original Source: N/A
    """
    sample_size = int(sample_size)
    k_variables = int(k_variables)
    return 1 - (1-r_square(beta=beta, sigma_market=sigma_market, sigma_epsilon=sigma_epsilon)) * (sample_size-1) / (sample_size-k_variables-1)



class CAPM:
    """
    ## Description
    CAPM, three-factor and five-factor Famma-French models.\n
    Contains methods for finding asset beta and predicting expected asset returns with the given beta.
    ### Input:
        - capm_params (CAPMParams): Parameters for defining a CAPM model instance
    """
    def __init__(self, capm_params: CAPMParams) -> None:
        self.capm_type = capm_params.capm_type
        self.solution_type = capm_params.solution_type
        self.market_returns = capm_params.market_returns
        self.rf = capm_params.rf
        self.smb = capm_params.smb
        self.hml = capm_params.hml
        self.rmw = capm_params.rmw
        self.cma = capm_params.cma
    
    def __validate_params(self) -> None:
        """
        ## Description
        Validation of paramers and their lengths.
        """
        match self.capm_type:
            case CAPMType.STANDARD:
                pass
            case CAPMType.THREE_FACTOR_FAMA_FRENCH:
                compare_array_len(array_1=self.market_returns, array_2=self.smb, array_1_name="market_returns", array_2_name="smb")
                compare_array_len(array_1=self.market_returns, array_2=self.hml, array_1_name="market_returns", array_2_name="hml")
            case CAPMType.FIVE_FACTOR_FAMA_FRENCH:
                compare_array_len(array_1=self.market_returns, array_2=self.smb, array_1_name="market_returns", array_2_name="smb")
                compare_array_len(array_1=self.market_returns, array_2=self.hml, array_1_name="market_returns", array_2_name="hml")
                compare_array_len(array_1=self.market_returns, array_2=self.rmw, array_1_name="market_returns", array_2_name="rmw")
                compare_array_len(array_1=self.market_returns, array_2=self.cma, array_1_name="market_returns", array_2_name="cma")

    
    def linear_regression(self, alpha: float, beta: float, beta_s: float=0.0, beta_h: float=0.0, beta_r: float=0.0, beta_c: float=0.0) -> np.ndarray:
        """
        ## Description
        Computes the expected return of an asset/project given the risk-free rate, expected market return, SMB, HML, RMW, CMA and their betas.
        ### Input:
            - alpha (float): y-axis intersection of the CAPM model
            - beta (float): Sensitivity of the asset with respect to premium market returns
            - beta_s (float): Sensitivity of the asset with respect to SMB returns
            - beta_h (float): Sensitivity of the asset with respect to HML returns
            - beta_r (float): Sensitivity of the asset with respect to RMW returns
            - beta_c (float): Sensitivity of the asset with respect to CMA returns
        ### Output:
            - Array of asset returns (np.ndarray)
        ### LaTeX Formula:
            - E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Linear_regression
            - Original Source: N/A
        """
        # Arguments validation
        self.__validate_params()
        # Linear regression
        lin_reg = float(alpha) + float(beta)*(self.market_returns - self.rf)
        if self.capm_type!=CAPMType.STANDARD:
            lin_reg = lin_reg + float(beta_s)*self.smb + float(beta_h)*self.hml
        if self.capm_type==CAPMType.FIVE_FACTOR_FAMA_FRENCH:
            lin_reg = lin_reg + float(beta_r)*self.rmw + float(beta_c)*self.cma
        return lin_reg
    
    def __linear_regression_wrapper(self, x: np.ndarray[Any, np.ndarray], alpha: float, beta: float, beta_s: float=0.0, beta_h: float=0.0,
                                    beta_r: float=0.0, beta_c: float=0.0) -> np.ndarray:
        """
        ## Description
        Wrapper function used for training linear regression.
        """
        self.market_returns = x[0]
        self.rf = x[1]
        match self.capm_type:
            case CAPMType.STANDARD:
                pass
            case CAPMType.THREE_FACTOR_FAMA_FRENCH:
                self.smb = x[2]
                self.hml = x[3]
            case CAPMType.FIVE_FACTOR_FAMA_FRENCH:
                self.smb = x[2]
                self.hml = x[3]
                self.rmw = x[4]
                self.cma = x[5]
        return self.linear_regression(alpha=alpha, beta=beta, beta_s=beta_s, beta_h=beta_h, beta_r=beta_r, beta_c=beta_c)

    
    def linear_regression_train(self, asset_returns: np.ndarray) -> dict[str, float]:
        """
        ## Description
        Finds the values of parameters alpha and betas (if COVARIANCE solution type is used, only beta is returned).
        ### Input:
            - asset_returns (np.ndarray): Array of asset returns
        ### Output:
            - alpha (float): y-axis intersection of the CAPM model
            - beta (float): Sensitivity of the asset with respect to premium market returns
            - beta_s (float): Sensitivity of the asset with respect to SMB returns
            - beta_h (float): Sensitivity of the asset with respect to HML returns
            - beta_r (float): Sensitivity of the asset with respect to RMW returns
            - beta_c (float): Sensitivity of the asset with respect to CMA returns
        """
        # Arguments validation
        self.__validate_params()
        compare_array_len(array_1=asset_returns, array_2=self.market_returns, array_1_name="asset_returns", array_2_name="market_returns")
        # Covariance solution
        if self.solution_type==CAPMSolutionType.COVARIANCE:
            cov_matrix = np.cov(asset_returns, self.market_returns, ddof=0)
            return {"beta":float(cov_matrix[1, 0]/cov_matrix[1,1])}
        # Linear regression solution
        else:
            ydata = scipy.array(asset_returns)
            match self.capm_type:
                case CAPMType.STANDARD:
                    xdata = [self.market_returns, self.rf]
                    def line(x, alpha, beta):
                        return self.__linear_regression_wrapper(x=x, alpha=alpha, beta=beta)
                    popt, _ = scipy.optimize.curve_fit(line, np.array(xdata), ydata)
                    return {"alpha":popt[0], "beta":popt[1]}
                case CAPMType.THREE_FACTOR_FAMA_FRENCH:
                    xdata = [self.market_returns, self.rf, self.smb, self.hml]
                    def line(x, alpha, beta, beta_s, beta_h):
                        return self.__linear_regression_wrapper(x=x, alpha=alpha, beta=beta, beta_s=beta_s, beta_h=beta_h)
                    popt, _ = scipy.optimize.curve_fit(line, np.array(xdata), ydata)
                    return {"alpha":popt[0], "beta":popt[1], "beta_s":popt[2], "beta_h":popt[3]}
                case CAPMType.FIVE_FACTOR_FAMA_FRENCH:
                    xdata = [self.market_returns, self.rf, self.smb, self.hml, self.rmw, self.cma]
                    def line(x, alpha, beta, beta_s, beta_h, beta_r, beta_c):
                        return self.__linear_regression_wrapper(x=x, alpha=alpha, beta=beta, beta_s=beta_s, beta_h=beta_h, beta_r=beta_r, beta_c=beta_c)
                    popt, _ = scipy.optimize.curve_fit(line, np.array(xdata), ydata)
                    return {"alpha":popt[0], "beta":popt[1], "beta_s":popt[2], "beta_h":popt[3], "beta_r":popt[4], "beta_c":popt[5]}
            