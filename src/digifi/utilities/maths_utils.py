
import numpy as np

# TODO: Add error function (SciPy)
# TODO: Add numerical solver to replace fsolve (SciPy)
# TODO: Add incomplete beta function
# TODO: Add minimize (SciPy)

PI = 3.141592653589793

def factorial(n: int) -> int:
    """
    Factorial of n defined through a recursion.
    """
    n = int(n)
    if n==0:
        return 1
    return n*factorial(n=n-1)

def sqrt(n: float) -> float:
    """
    Square root of n.
    """
    n = float(n)
    return n**0.5


def n_choose_r(n: int, r: int) -> int:
    """
    nCr: n choose r
    """
    n = int(n)
    r = int(r)
    return factorial(n=n)/(factorial(n=n-r)*factorial(n=r))

def erf(n: float, terms: int=20) -> float:
    """
    Taylor series expansion to the 20th (tested for the optimal value)
    """
    total = 0
    for n in range(terms):
        total += ((-1)**n) * (n**(2*n + 1)) / (factorial(n) * (2*n + 1))
    return (2 / sqrt(PI)) * total

def numerical_solver():
    """
    Numerical solver
    https://math.stackexchange.com/questions/3642041/what-is-the-function-fsolve-in-python-doing-mathematically
    https://github.com/scipy/scipy/blob/d0a431ac40a47fa849a9db5884b5e5b88069f5ee/scipy/optimize/minpack.py#L46
    extremely hard to imeplement without the actual source code because the MINPACK's algortihm is very complex which
    would take too much time to implement. My suggestion is to import scipy partially which could help with other functions as well
    """


def incomplete_beta_function():
    """
    Incomplete beta function, continued fraction expansion
    https://dlmf.nist.gov/8.17#E24
    """
    


def minimize():
    """
    Minimize
    """