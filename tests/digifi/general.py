import pandas as pd



def get_test_stock_data() -> pd.DataFrame:
    """
    Import test_stock_data.csv file.
    """
    return pd.read_csv(r"./tests/digifi/test_stock_data.csv", index_col="Date")