from typing import Union
import numpy as np



def factorial(n: int) -> int:
    """
    ## Description
    Factorial of n defined through a recursion.
    ## Links
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
    ## Links
    - Wikipedia: https://en.wikipedia.org/wiki/Binomial_coefficient
    - Original Source: N/A
    """
    n = int(n)
    r = int(r)
    return factorial(n=n)/(factorial(n=n-r)*factorial(n=r))



def erf(x: Union[np.ndarray, float], n_terms: int=20) -> Union[np.ndarray, float]:
    """
    ## Description
    Error function computed with the Taylor expansion.
    ## Links
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