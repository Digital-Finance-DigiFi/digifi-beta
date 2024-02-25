import numpy as np
import digifi as dgf
import tests as dgf_tests



class TestRiskMeasures:
    """
    ## Description
    Test RiskMeasures class.
    """
    def __init__(self) -> None:
        self.alpha = 0.05
        self.returns_distribution = dgf.NormalDistribution(mu=0.0, sigma=0.0264)
    
    def integration_test_value_at_risk(self) -> float:
        return dgf.RiskMeasures().value_at_risk(alpha=self.alpha, returns_distribution=self.returns_distribution)
    
    def integration_test_expected_shortfall(self) -> float:
        return dgf.RiskMeasures().expected_shortfall(alpha=self.alpha, returns_distribution=self.returns_distribution)



class TestPortfolio:
    """
    ## Description
    Test Portfolio class.
    """
    def integration_test_efficient_frontier(self) -> dict[str, dict]:
        # Sample data
        data = dgf_tests.get_test_portfolio_data()
        stock_list = list(data.keys())
        stock_list.remove("Date")
        # Inputs definition
        weights = np.ones(len(stock_list))/len(stock_list)
        assets = dict()
        predictable_income = dict()
        for column in stock_list:
            assets[column] = data[column]
            predictable_income[column] = 0.05*np.ones(len(data[column]))
        # Integration test
        portfolio = dgf.Portfolio(assets=assets, weights=weights, predictable_income=predictable_income)
        return portfolio.efficient_frontier(r=0.02, frontier_n_points=50)



class TestInstrumentsPortfolio:
    """
    ## Description
    Test InstrumentsPortfolio class.
    """
    def integration_test_efficient_frontier(self) -> dict[str, dict]:
        n_steps = 100
        T = 5.0
        r = 0.02
        t = np.arange(start=0, stop=T, step=T/n_steps)
        # Sample data
        bond_principal = 100.0
        bond_coupon_rate = 0.05
        bond_prices = dgf.GeometricBrownianMotion(mu=0.01, sigma=0.05, n_paths=1, n_steps=n_steps-1, T=T, s_0=bond_principal).get_paths()[0]
        bond_coupons = np.zeros(n_steps)
        bond_coupons[40] = bond_coupon_rate * bond_principal
        option_initial_price = 6.0
        option_prices = dgf.GeometricBrownianMotion(mu=0.0, sigma=0.3, n_paths=1, n_steps=n_steps-1, T=T, s_0=option_initial_price).get_paths()[0]
        option_predictable_income = np.zeros(n_steps)
        # Instuments definitions
        bond_struct = dgf.BondStruct(bond_type=dgf.BondType.ANNUITY_BOND, principal=bond_principal, coupon_rate=bond_coupon_rate,
                                     discount_rate=r, maturity=T, initial_price=bond_principal)
        bond = dgf.Bond(bond_struct=bond_struct,
                        financial_instrument_struct=dgf.FinancialInstrumentStruct(instrument_type=dgf.FinancialInstrumentType.CASH_INSTRUMENT,
                                                                                  asset_class=dgf.FinancialInstrumentAssetClass.DEBT_BASED_INSTRUMENT,
                                                                                  identifier="bond"),
                        portfolio_instrument_struct=dgf.PortfolioInstrumentStruct(portfolio_price_array=bond_prices,
                                                                                  portfolio_time_array=t,
                                                                                  portfolio_predicatble_income=bond_coupons))
        option_struct = dgf.OptionStruct(asset_price=50.0, strike_price=50.0, discount_rate=r, dividend_yield=0.0, time_to_maturity=T,
                                         sigma=0.2, initial_option_price=option_initial_price, option_type=dgf.OptionType.EUROPEAN,
                                         payoff_type=dgf.OptionPayoffType.CALL)
        option = dgf.Option(option_struct=option_struct, financial_instrument_struct=dgf.FinancialInstrumentStruct(instrument_type=dgf.FinancialInstrumentType.DERIVATIVE_INSTRUMENT,
                                                                                                                   asset_class=dgf.FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                                   identifier="option"),
                            portfolio_instrument_struct=dgf.PortfolioInstrumentStruct(portfolio_price_array=option_prices,
                                                                                      portfolio_time_array=t,
                                                                                      portfolio_predicatble_income=option_predictable_income))
        # Integration test
        portfolio = dgf.InstrumentsPortfolio(instruments=[bond, option], weights=np.array([0.5, 0.5]))
        return portfolio.efficient_frontier(r=r, frontier_n_points=30)