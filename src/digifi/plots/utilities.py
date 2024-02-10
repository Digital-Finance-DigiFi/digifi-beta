from typing import Any
import numpy as np



def type_check(value: Any, type_: type[Any], value_name: str) -> None:
    """
    Perform dynamic type check for a value to be of a defined type.
    """
    if isinstance(value, type_) is False:
        raise TypeError("The argument {} must be of {} type.".format(str(value_name), type_))



def compare_array_len(array_1: np.ndarray, array_2: np.ndarray, array_1_name: str="array_1", array_2_name: str="array_2") -> None:
    """
    Compare that the two arrays provided are of the same length, while also verifying that both arrays are of np.ndarray type.
    """
    type_check(value=array_1, type_=np.ndarray, value_name=str(array_1_name))
    type_check(value=array_2, type_=np.ndarray, value_name=str(array_2_name))
    if len(array_1)!=len(array_2):
        raise ValueError("The length of {0} and {1} do not coincide.".format(array_1_name, array_2_name))