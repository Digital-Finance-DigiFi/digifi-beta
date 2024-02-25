import digifi as dgf
import tests as dgf_tests



class TestCAPM:
    """
    ## Description
    Test CAPM class.
    """
    def __init__(self) -> None:
        capm_data = dgf_tests.get_test_capm_data()
        capm_params = dgf.CAPMParams(capm_type=dgf.CAPMType.FIVE_FACTOR_FAMA_FRENCH, solution_type=dgf.CAPMSolutionType.LINEAR_REGRESSION,
                                     market_returns=capm_data["Mkt"], rf=capm_data["RF"], smb=capm_data["SMB"], hml=capm_data["HML"],
                                     rmw=capm_data["RMW"], cma=capm_data["CMA"])
        self.asset_returns = capm_data["AAPL"]
        self.capm_params = capm_params
        self.capm = dgf.CAPM(capm_params=capm_params)
    
    def integration_test_linear_regression_train(self) -> dict:
        return self.capm.linear_regression_train(asset_returns=self.asset_returns)