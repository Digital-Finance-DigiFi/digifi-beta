import numpy as np
import digifi as dgf



class TestBond:
    """
    ## Description
    Test Bond class.
    """
    def __init__(self) -> None:
        bond_struct = dgf.BondStruct(bond_type=dgf.BondType.ANNUITY_BOND, principal=100.0, coupon_rate=0.05, discount_rate=0.04,
                                     maturity=3.0, initial_price=98.0)
        self.bond_struct = bond_struct
        self.bond = dgf.Bond(bond_struct=bond_struct)
    
    def integration_test_present_value(self) -> float:
        return self.bond.present_value()



class TestOption:
    """
    ## Description
    Test Option class.
    """
    def __init__(self) -> None:
        option_struct = dgf.OptionStruct(asset_price=49.0, strike_price=50.0, discount_rate=0.02, dividend_yield=0.0,
                                         time_to_maturity=1.0, sigma=0.2, initial_option_price=3.9, option_type=dgf.OptionType.EUROPEAN,
                                         payoff_type=dgf.OptionPayoffType.LONG_CALL)
        self.option_struct = option_struct
        self.option = dgf.Option(option_struct=option_struct, option_pricing_method=dgf.OptionPricingMethod.BINOMIAL, payoff=None)
    
    def integration_test_present_value(self) -> float:
        return self.option.present_value(lattice_model_n_steps=50)
    
    def integration_test_present_value_surface(self) -> dict:
        return self.option.present_value_surface(start_price=20.0, stop_price=80.0, n_prices=100, n_timesteps=30, lattice_model_n_steps=20)



class TestStock:
    """
    ## Description
    Test Stock class.
    """
    def __init__(self) -> None:
        stock_struct = dgf.StockStruct(price_per_share=50.0, n_shares_outstanding=1_000, dividend_per_share=2.0, earnings_per_share=1.2,
                                       quote_values=dgf.QuoteValues.PER_SHARE, dividend_growth_rate=0.0, dividend_compounding_frequency=1.0,
                                       initial_price=49.0)
        self.stock_struct = stock_struct
        self.stock = dgf.Stock(stock_struct=stock_struct)
    
    def integration_test_present_value(self, method: dgf.StockValuationType) -> float:
        dgf.type_check(value=method, type_=dgf.StockValuationType, value_name="method")
        expected_dividend = 3.0
        pe, pb, ev_to_ebitda = 11.0, 3.0, 6.0
        valuation_params = dgf.ValuationByComparablesParams(valuations=np.array([60_000, 55_000, 54_000, 40_000, 56_000]),
                                                            pe_ratios=np.array([10.0, 16.0, 14.0, 25.0, 14.5]),
                                                            pb_ratios=np.array([5.0, 3.0, 4.5, 3.0, 4.0]),
                                                            ev_to_ebitda=np.array([10.0, 7.0, 8.0, 10.0, 6.0]))
        return self.stock.present_value(stock_valuation_method=method, expected_dividend=expected_dividend, pe=pe, pb=pb,
                                        ev_to_ebitda=ev_to_ebitda, valuation_params=valuation_params)