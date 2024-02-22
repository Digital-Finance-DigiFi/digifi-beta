from typing import Any
import numpy as np



def compare_array_len(array_1: np.ndarray, array_2: np.ndarray, array_1_name: str="array_1", array_2_name: str="array_2") -> None:
    """
    ## Description
    Asserts that the two arrays provided are of the same length, while also verifying that both arrays are of np.ndarray type.
    ### Input:
        - array_1 (np.ndarray): First array
        - array_2 (np.ndarray): Second array
        - array_1_name (str): Name of the first array, which will be printed in case of a TypeError
        - array_1_name (str): Name of the second array, which will be printed in case of a TypeError
    """
    type_check(value=array_1, type_=np.ndarray, value_name=array_1_name)
    type_check(value=array_2, type_=np.ndarray, value_name=array_2_name)
    if len(array_1)!=len(array_2):
        raise ValueError("The length of {0} and {1} do not coincide.".format(array_1_name, array_2_name))



def rolling(array: np.ndarray, window: int) -> np.ndarray:
    """
    ## Description
    Rolling window over an array.
    ### Input:
        - array (np.ndarray): Array over which the rolling window is taken
        - window (int): Size of the rolling window
    """
    window = int(window)
    type_check(value=array, type_=np.ndarray, value_name="array")
    shape = array.shape[:-1] + (array.shape[-1]-window+1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)



def type_check(value: Any, type_: type[Any], value_name: str) -> None:
    """
    ## Description
    Perform dynamic type check for a value to be of the expected type.
    ### Input:
        - value (Any): Instance that will be checked
        - type_ (type[Any]): The type that instance is expected to be
        - value_name (str): Name of the value, which will be printed in case of a TypeError
    """
    if isinstance(value, type_) is False:
        raise TypeError("The argument {} must be of {} type.".format(str(value_name), type_))



class DataClassValidation:
    """
    ## Description
    Validate fields and types in dataclass instances.\n
    If dataclass instance has custom validation methods in the format validate_<field_name>,
    where <field_name> is replaced with the name of the field - the custom validation method will be run.
    """
    def __post_init__(self) -> None:
        # Validate types
        for k, v in self.__annotations__.items():
            try:
                assert isinstance(getattr(self, k), v)
            except AssertionError:
                raise TypeError("The argument {} must be of {} type.".format(k, v))
        # Apply custom validation methods inside dataclass definition
        for name, field in self.__dataclass_fields__.items():
            if (method := getattr(self, f"validate_{name}", None)):
                setattr(self, name, method(getattr(self, name), field=field))
