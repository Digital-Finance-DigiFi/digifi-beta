from typing import Union, Tuple
import numpy as np
import pandas as pd



def verify_array(array: np.ndarray, array_name: str="price_array") -> None:
    """
    Verify that the array provided is a numpy.ndarray array.
    """
    if isinstance(array, np.ndarray)==False:
        raise TypeError("The argument {} must be of numpy.ndarray type.".format(array_name))



def compare_array_len(array_1: np.ndarray, array_2: np.ndarray, array_1_name: str="array_1", array_2_name: str="array_2") -> None:
    """
    Compare that the two arrays provided are of the same length, while also verifying that both arrays are of numpy.ndarray type.
    """
    verify_array(array_1)
    verify_array(array_2)
    if len(array_1)!=len(array_2):
        raise ValueError("The length of {0} and {1} do not coincide.".format(array_1_name, array_2_name))



def generate_ohlc_price_df(open_price: np.ndarray, high_price: np.ndarray, low_price: np.ndarray, close_price: np.ndarray,
                           timestamp: np.ndarray, volume: Union[np.ndarray, None]=None) -> pd.DataFrame:
    """
    Generate a pandas.DataFrame with columns Open, High, Low, Close and Volume based on the provided arrays.
    """
    compare_array_len(array_1=close_price, array_2=open_price, array_1_name="close_price", array_2_name="open_price")
    compare_array_len(array_1=close_price, array_2=high_price, array_1_name="close_price", array_2_name="high_price")
    compare_array_len(array_1=close_price, array_2=low_price, array_1_name="close_price", array_2_name="low_price")
    compare_array_len(array_1=close_price, array_2=timestamp, array_1_name="close_price", array_2_name="timestamp")
    price_df = pd.concat((pd.Series(open_price, name="Open"), pd.Series(high_price, name="High"), pd.Series(low_price, name="Low"),
                          pd.Series(close_price, name="Close")), axis=1)
    price_df.index = timestamp
    if isinstance(volume, np.ndarray):
        compare_array_len(array_1=close_price, array_2=volume, array_1_name="close_price", array_2_name="volume")
        price_df = pd.concat((price_df, pd.Series(volume, index=timestamp)), axis=1, ignore_index=True)
        price_df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return price_df
    