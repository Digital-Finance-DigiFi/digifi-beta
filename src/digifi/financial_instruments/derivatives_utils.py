from typing import Type
import abc
import numpy as np
from digifi.utilities.general_utils import type_check



class OptionPayoffs:
    """
    ## Description
    Common option payoff functions.
    """

    @staticmethod
    def long_call_payoff(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Long call option payoff.
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Option strike price
        ### Output:
            - Payoff (np.ndarray) of the long call option
        ### LaTeX Formula:
            - \\textit{Long Call Payoff} = max(S_{t}-K, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-float(k), np.zeros(len(s_t)))
    
    @staticmethod
    def short_call_payoff(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Short call option payoff.
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Option strike price
        ### Output:
            - Payoff (np.ndarray) of the short call option
        ### LaTeX Formula:
            - \\textit{Short Call Payoff} = min(K-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-float(k), np.zeros(len(s_t)))
    
    @staticmethod
    def long_put_payoff(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Long put option payoff.
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Option strike price
        ### Output:
            - Payoff (np.ndarray) of the long put option
        ### LaTeX Formula:
            - \\textit{Long Put Payoff} = max(K-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(float(k)-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def short_put_payoff(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        LonShort put option payoff.
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Option strike price
        ### Output:
            - Payoff (np.ndarray) of the short put option
        ### LaTeX Formula:
            - \\textit{Short Put Payoff} = min(S_{t}-K, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(float(k)-s_t, np.zeros(len(s_t)))

    @staticmethod
    def bull_collar_payoff(s_t: np.ndarray, k_p: float=0.0, k_c: float=0.0) -> np.ndarray:
        """
        ## Description
        Bull collar option payoff.\n
        Bull Collar = Asset + Long Put + Short Call.
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_p (float): Long put option strike price
            - k_c (float): Short call option strike price
        ### Output:
            - Payoff (np.ndarray) of the bull collar option
        ### LaTeX Formula:
            - \\textit{Bull Collar Payoff} = S_{t} + max(K_{p}-S_{t}, 0) + min(K_{c}-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return s_t + np.maximum(float(k_p)-s_t, np.zeros(len(s_t))) + np.minimum(float(k_c)-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def bear_collar_payoff(s_t: np.ndarray, k_p: float=0.0, k_c: float=0.0) -> np.ndarray:
        """
        ## Description
        Bear collar option payoff.\n
        Bear Collar = - Asset + Short Put + Long Call
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_p (float): Long call option strike price
            - k_c (float): short put option strike price
        ### Output:
            - Payoff (np.ndarray) of the bear collar option
        ### LaTeX Formula:
            - \\textit{Bear Collar Payoff} = - S_{t} + min(S_{t}-K_{p}, 0) + max(S_{t}-K_{c}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return -s_t + np.minimum(s_t-float(k_p), np.zeros(len(s_t))) + np.maximum(s_t-float(k_c), np.zeros(len(s_t)))
    
    @staticmethod
    def bull_spread(s_t: np.ndarray, k_l: float=0.0, k_s: float=0.0) -> np.ndarray:
        """
        ## Description
        Bull spread option payoff.\n
        Bull Spread = Long Call + Short Call
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_l (float): Long call option strike price
            - k_s (float): Short call option strike price
        ### Output:
            - Payoff (np.ndarray) of the bull spread option
        ### LaTeX Formula:
            - \\textit{Bull Spread Payoff} = max(S_{t}-K_{l}, 0) + min(K_{s}-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-float(k_l), np.zeros(len(s_t))) + np.minimum(float(k_s)-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def bear_spread(s_t: np.ndarray, k_l: float=0.0, k_s: float=0.0) -> np.ndarray:
        """
        ## Description
        Bear spread option payoff.\n
        Bear Spread = Long Put + Short Put
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_l (float): Long put option strike price
            - k_s (float): Short put option strike price
        ### Output:
            - Payoff (np.ndarray) of the bear spread option
        ### LaTeX Formula:
            - \\textit{Bull Spread Payoff} = max(K_{c}-S_{t}, 0) + min(S_{t}-K_{p}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(float(k_l)-s_t, np.zeros(len(s_t))) + np.minimum(s_t-float(k_s), np.zeros(len(s_t)))
    
    @staticmethod
    def long_butterfly(s_t: np.ndarray, k_1: float=0.0, k: float=0.0, k_2: float=0.0) -> np.ndarray:
        """
        ## Description
        Butterfly spread option payoff.\n
        Buttefly Spread = Long Call + 2*Short Put + Long Call
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_1 (float): Smaller long call option strike price
            - k (float): Short put option strike price
            - k_2 (float): Larger long call option strike price
        ### Output:
            - Payoff (np.ndarray) of the long butterfly option
        ### LaTeX Formula:
            - \\textit{Butterfly Spread Payoff} = max(S_{t}-K_{1}, 0) + min(K-S_{t}, 0) + max(S_{t}-K_{2}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return (np.maximum(s_t-float(k_1), np.zeros(len(s_t)))
                + np.minimum(float(k)-s_t, np.zeros(len(s_t)))
                + np.maximum(s_t-float(k_2), np.zeros(len(s_t))))
    
    @staticmethod
    def box_spread(s_t: np.ndarray, k_1: float=0.0, k_2: float=0.0) -> np.ndarray:
        """
        ## Description
        Box spread option payoff.\n
        Box Spread = Long Call + Short Call + Long Put + Short Put = Bull Spread + Bear Spread
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_1 (float): Smaller option strike price
            - k_2 (float): Larger option strike price
        ### Output:
            - Payoff (np.ndarray) of the box spread option
        ### LaTeX Formula:
            - \\textit{Box Spread Payoff} = max(S_{t}-K_{1}, 0) + min(K_{2}-S_{t}, 0) + max(K_{1}-S_{t}, 0) + min(S_{t}-K_{2}, 0)
        """
        k_1 = float(k_1)
        k_2 = float(k_2)
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return (np.maximum(s_t-k_1, np.zeros(len(s_t))) + np.minimum(k_2-s_t, np.zeros(len(s_t)))
                + np.maximum(k_1-s_t, np.zeros(len(s_t))) + np.minimum(s_t-k_2, np.zeros(len(s_t))))
    
    @staticmethod
    def straddle(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Straddle option payoff.\n
        Straddle = Long Call + Long Put
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Straddle option strike price
        ### Output:
            - Payoff (np.ndarray) of the straddle option
        ### LaTeX Formula:
            - \\textit{Straddle Payoff} = max(S_{t}-K, 0) + max(K-S_{t}, 0)
        """
        k = float(k)
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-k, np.zeros(len(s_t))) + np.maximum(k-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def strangle(s_t: np.ndarray, k_1: float=0.0, k_2: float=0.0) -> np.ndarray:
        """
        ## Description
        Strangle option payoff.\n
        Strangle = Long Call + Long Put
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k_1 (float): Smaller stangle option strike price
            - k_2 (float): Larger strangle option strike price
        ### Output:
            - Payoff (np.ndarray) of the strangle option
        ### LaTeX Formula:
            - \\textit{Strangle Payoff} = max(S_{t}-K_{2}, 0) + max(K_{1}-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-float(k_2), np.zeros(len(s_t))) + np.maximum(float(k_1)-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def strip(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Strip option payoff.\n
        Strip = Long Call + 2*Long Put
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Strip option strike price
        ### Output:
            - Payoff (np.ndarray) of the strip option
        ### LaTeX Formula:
            - \\textit{Strip Payoff} = max(S_{t}-K, 0) + 2max(K-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return np.maximum(s_t-float(k), np.zeros(len(s_t))) + 2*np.maximum(float(k)-s_t, np.zeros(len(s_t)))
    
    @staticmethod
    def strap(s_t: np.ndarray, k: float=0.0) -> np.ndarray:
        """
        ## Description
        Strap option payoff.\n
        Strap = 2*Long Call + Long Put
        ### Input:
            - s_t (np.ndarray): Underlying asset price
            - k (float): Strap option strike price
        ### Output:
            - Payoff (np.ndarray) of the strap option
        ### LaTeX Formula:
            - \\textit{Strap Payoff} = 2max(S_{t}-K, 0) + max(K-S_{t}, 0)
        """
        type_check(value=s_t, type_=np.ndarray, value_name="s_t")
        return 2*np.maximum(s_t-float(k), np.zeros(len(s_t))) + np.maximum(float(k)-s_t, np.zeros(len(s_t)))



class CustomPayoff(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "payoff") and
                callable(subclass.payoff))
    
    @abc.abstractmethod
    def payoff(self, s_t: np.ndarray) -> np.ndarray:
        """
        ## Description
        Payoff function.
        """
        raise NotImplementedError



class LongCallPayoff(CustomPayoff):
    """
    ## Description
    Helper class to define the long call payoff in library functions and methods.
    """
    def __init__(self, k: float) -> None:
        self.k = float(k)
    
    def payoff(self, s_t: np.ndarray) -> np.ndarray:
        return OptionPayoffs().long_call_payoff(s_t=s_t, k=self.k)



class LongPutPayoff(CustomPayoff):
    """
    ## Description
    Helper class to define the long put payoff in library functions and methods.
    """
    def __init__(self, k: float) -> None:
        self.k = float(k)
    
    def payoff(self, s_t: np.ndarray) -> np.ndarray:
        return OptionPayoffs().long_put_payoff(s_t=s_t, k=self.k)



def validate_custom_payoff(custom_payoff: Type[CustomPayoff], length_value: int=5) -> Type[CustomPayoff]:
    """
    ## Description
    Validate custom payoff object to satisfy the computational requirements.
    ### Input:
        - custom_payoff (Type[CustomPayoff]): Custom payoff class to be validated
        - length_value (int): Number of test data points to validate payoff method on
    ### Output:
        - Validated custom payoff class (Type[CustomPayoff])
    """
    length_value = int(length_value)
    # Check that custom_payoff takes in parameter s_t
    s_t = np.ones(length_value)
    result = custom_payoff.payoff(s_t=s_t)
    # Check that custom_payoff produces an array of length_value
    type_check(value=result, type_=np.ndarray, value_name="result")
    if len(result)!=length_value:
        raise ValueError("The custom payoff does not produce an array of length {}.".format(length_value))
    return custom_payoff