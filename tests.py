# import sys
# sys.path.append("../src")
import numpy as np
import src.digifi as dgf



def main():
    dgf.StandardNormalZigguratAlgorithmPseudoRandomNumberGenerator(sample_size=10_000).plot_3d_scattered_points()



if __name__=="__main__":
    # test_df = pd.read_csv(r"test_stock_data.csv", index_col="Date")
    main()
