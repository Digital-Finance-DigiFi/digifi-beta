# import sys
# sys.path.append("../src")
import numpy as np
import src.digifi as dgf

import datetime
import pandas as pd
import yfinance as yf



def main():
    stock_data_df: pd.DataFrame = yf.download(["JPM", "GS", "BAC", "C", "WFC", "BCS", "HSBC"],
                                              start=datetime.datetime.now()-datetime.timedelta(days=365),
                                              end=datetime.datetime.now())["Adj Close"]
    assets = dict()
    for ticker in stock_data_df.columns:
        assets[ticker] = stock_data_df[ticker].to_numpy()
    portfolio = dgf.Portfolio(assets=assets, weights=np.array([0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]))
    portfolio.plot_efficient_frontier(r=0.03, frontier_n_points=50)



if __name__=="__main__":
    # test_df = pd.read_csv(r"test_stock_data.csv", index_col="Date")
    main()
