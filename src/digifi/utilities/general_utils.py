import numpy as np



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



def rolling(array: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling window over an array.
    """
    window = int(window)
    verify_array(array=array, array_name="array")
    shape = array.shape[:-1] + (array.shape[-1]-window+1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    