from typing import Union
import numpy as np



def factorial(n: int) -> int:
    """
    ## Description
    Factorial of n defined through a recursion.
    ### Input:
        - n (int): Input variable
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Factorial
        - Original Source: N/A
    """
    n = int(n)
    if n==0:
        return 1
    return n*factorial(n=n-1)



def n_choose_r(n: int, r: int) -> int:
    """
    ## Description
    nCr: n choose r
    ### Input:
        - n (int): Power of the binomial expansion
        - r (int): Number of successes
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Binomial_coefficient
        - Original Source: N/A
    """
    n = int(n)
    r = int(r)
    if n<r:
        raise ValueError("The value of variable n must be larger or equal to the value of variable r")
    return factorial(n=n)/(factorial(n=n-r)*factorial(n=r))



def erf(x: Union[np.ndarray, float], n_terms: int=20) -> Union[np.ndarray, float]:
    """
    ## Description
    Error function computed with the Taylor expansion.
    ### Input:
        - x (Union[np.ndarray, float]): Input variables
        - n_terms (int): Number of terms to use in a Taylor's expansion of the error function
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Error_function
        - Original Source: N/A
    """
    # Arguments validation
    if isinstance(x, np.ndarray):
        total = np.zeros(len(x))
    else:
        x = float(x)
        total = 0.0
    # Taylor expansion of the error function
    for n in range(int(n_terms)):
        total += ((-1)**n) * (x**(2*n + 1)) / (factorial(n) * (2*n + 1))
    return (2 / np.sqrt(np.pi)) * total