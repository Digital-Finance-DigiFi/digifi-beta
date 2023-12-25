import numpy as np
# TODO: Add error function
# TODO: Add numerical solver to replace fsolve
# TODO: Add incomplete beta function



def factorial(n: int) -> int:
    """
    Factorial of n defined through a recursion.
    """
    n = int(n)
    if n==0:
        return 1
    return n*factorial(n=n-1)



def n_choose_r(n: int, r: int) -> int:
    """
    nCr: n choose r
    """
    n = int(n)
    r = int(r)
    return factorial(n=n)/(factorial(n=n-r)*factorial(n=r))