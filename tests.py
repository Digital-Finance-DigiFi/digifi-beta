# import sys
# sys.path.append("../src")
import numpy as np
import src.digifi as dgf



def main():
    trinomial_model = dgf.BrownianMotionTrinomialModel(s_0=102, k=100, T=2, r=0.03, sigma=0.15, q=0.02, n_steps=100,
                                                       payoff_type=dgf.LatticeModelPayoffType.PUT)
    print("Trinomial: ", trinomial_model.european_option_trinomial_model())
    binomial_model = dgf.BrowninMotionBinomialModel(s_0=102, k=100, T=2, r=0.03, sigma=0.15, q=0.02, n_steps=100,
                                                       payoff_type=dgf.LatticeModelPayoffType.PUT)
    print("Binomial: ", binomial_model.european_option_binomial_model())


if __name__=="__main__":
    # test_df = pd.read_csv(r"test_stock_data.csv", index_col="Date")
    main()
