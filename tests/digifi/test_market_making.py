import digifi as dgf



class TestSimpleAMM:
    """
    ## Description
    Test SimpleAMM class.
    """
    def integration_test_make_transaction(self) -> dict[str, float]:
        # Inputs definition
        token_1 = dgf.AMMToken(identifier="BTC", supply=10.0, fee_lower_bound=0.0, fee_upper_bound=0.03)
        token_2 = dgf.AMMToken(identifier="ETH", supply=1_000.0, fee_lower_bound=0.0, fee_upper_bound=0.03)
        liquidity_pool = dgf.AMMLiquidityPool(token_1=token_1, token_2=token_2, char_number=10_000)
        tx_data = dgf.AMMTransactionData(token_id="BTC", quantity=1.0, percent_fee=0.01)
        # Integration test
        simple_amm = dgf.SimpleAMM(initial_liquidity_pool=liquidity_pool)
        return simple_amm.make_transaction(tx_data=tx_data)