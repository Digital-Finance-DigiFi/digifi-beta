# import sys
# sys.path.append("../src")
import numpy as np
import pandas as pd
import src.digifi as df



def main():
    test_df = pd.read_csv(r"test_stock_data.csv", index_col="Date")
    fig = df.plot_candlestick_chart(open_price=np.array(test_df["Open"]), high_price=np.array(test_df["High"]), low_price=np.array(test_df["Low"]),
                              close_price=np.array(test_df["Close"]), timestamp=np.array(test_df.index), volume=np.array(test_df["Volume"]),
                              indicator_subplot=True, return_fig_object=True)
    df.plot_adx(fig=fig, high_price=np.array(test_df["High"]), low_price=np.array(test_df["Low"]), close_price=np.array(test_df["Close"]),
                timestamp=np.array(test_df.index))



if __name__=="__main__":
    main()