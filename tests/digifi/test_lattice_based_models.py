import digifi as dgf



class TestBrownianMotionBinomialModel:
    """
    ## Description
    Test BrownianMotionBinomialModel class.
    """
    def __init__(self) -> None:
        self.binomial_model = dgf.BrownianMotionBinomialModel(s_0=49.0, k=50.0, T=1.0, r=0.02, sigma=0.2, q=0.0, n_steps=10,
                                                              payoff_type=dgf.LatticeModelPayoffType.LONG_CALL, custom_payoff=None)

    def integration_test_european_option(self) -> float:
        return self.binomial_model.european_option()
    
    def integration_test_bermudan_option(self) -> float:
        return self.binomial_model.bermudan_option()



class TestBrownianMotionTrinomialModel:
    """
    ## Description
    Test BrownianMotionBinomialModel class.
    """
    def __init__(self) -> None:
        self.trinomial_model = dgf.BrownianMotionTrinomialModel(s_0=49.0, k=50.0, T=1.0, r=0.02, sigma=0.2, q=0.0, n_steps=10,
                                                                payoff_type=dgf.LatticeModelPayoffType.LONG_CALL, custom_payoff=None)

    def integration_test_european_option(self) -> float:
        return self.trinomial_model.european_option()
    
    def integration_test_bermudan_option(self) -> float:
        return self.trinomial_model.bermudan_option()