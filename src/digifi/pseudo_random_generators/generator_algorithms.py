from typing import Callable
import numpy as np
from digifi.utilities.general_utils import (compare_array_len, type_check)
from digifi.probability_distributions.continuous_probability_distributions import NormalDistribution
from digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator



def accept_reject_method(f_x: Callable, g_x: Callable, Y_sample: np.ndarray, M: float, uniform_sample: np.ndarray,
                         sample_size: int=10_000) -> np.ndarray:
    compare_array_len(array_1=Y_sample, array_2=uniform_sample, array_1_name="Y_sample", array_2_name="uniform_sample")
    """
    ## Description
    Implements the Accept-Reject Method, a Monte Carlo technique for generating random samples from a probability distribution.
    ### Input:
        - f_x (Callable): Target probability density function
        - g_x (Callable): Proposal probability density function
        - Y_sample (np.ndarray): Sample from the proposal distribution
        - M (float): Constant such that M*g(x) >= f(x) for all x
        - uniform_sample (np.ndarray): Sample from a uniform distribution
        - sample_size (int, optional): Size of the sample to generate
    ### Output:
        - Sample array (np.ndarray) generated from the target distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Rejection_sampling
        - Original Source: N/A
    """
    g = g_x(Y_sample)
    f = f_x(Y_sample)
    X_sample = np.array([])
    for j in range(sample_size):
        if (uniform_sample[j]*M*g[j])<=f[j]:
            X_sample = np.append(X_sample, Y_sample[j])
    return X_sample



def inverse_transform_method(f_x: Callable, sample_size: int=10_000, seed: int=12_345) -> np.ndarray:
    """
    ## Description
    Uses the Inverse Transform Method to generate random samples from a specified probability distribution function.
    ### Input:
        - f_x (Callable): Inverse of the cumulative distribution function
        - sample_size (int, optional): Number of random samples to generate
        - seed (int, optional): Seed for the pseudo-random number generator
    ### Output:
        - Sample array (np.ndarray) generated from the specified distribution
    ### Latex:
        - X = F^{-1}(U)
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Inverse_transform_sampling
        - Original SOurce: N/A
    """
    u = FibonacciPseudoRandomNumberGenerator(seed=seed, sample_size=sample_size).generate()
    return f_x(u)



def box_muller_algorithm(uniform_array_1: np.ndarray, uniform_array_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ## Description
    The Box-Muller algorithm transforms uniformly distributed samples into samples distributed according to the standard normal distribution.
    ### Input:
        - uniform_array_1 (np.ndarray): First array of uniform samples
        - uniform_array_2 (np.ndarray): Second array of uniform samples
    ### Output:
        - Two arrays (np.ndarrays), each containing normal distributed samples
    ### Latex:
        -  Z_{0} = \\sqrt{-2ln(U_{1})} \\cos(2\\pi U_{2})
        -  Z_{1} = \\sqrt{-2ln(U_{1})} \\sin(2\\pi U_{2})
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Box-Muller_transform
        - Original Source: https://doi.org/10.1214%2Faoms%2F1177706645
    """

    return (np.sqrt(-2*np.log(uniform_array_1))*np.cos(2*np.pi*uniform_array_2),
            np.sqrt(-2*np.log(uniform_array_1))*np.sin(2*np.pi*uniform_array_2))



def marsaglia_method(max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> tuple[float, float]:
    """
    ## Description
    The Marsaglia polar method for generating standard normal random variables from uniformly distributed random numbers.
    ### Input:
        - max_iterations (int, optional): Maximum number of iterations
        - seed_1 (int, optional): Seed for the first pseudo-random number generator
        - seed_2 (int, optional): Seed for the second pseudo-random number generator
    ### Output:
        - A pair of standard normal random variables (np.ndarray)
    ### Latex:
        - S = W^{2}_{1} + W^{2}_{2}
        - If S < 1, then:
        - Z_{0} = W_{1} \\sqrt{\\frac{-2ln(S)}{S}}
        - Z_{1} = W_{2} \\sqrt{\\frac{-2ln(S)}{S}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Marsaglia_polar_method
        - Original Source: https://doi.org/10.1137%2F1006063
    """
    w_1 = 2*FibonacciPseudoRandomNumberGenerator(seed=seed_1, sample_size=max_iterations).generate() - 1
    w_2 = 2*FibonacciPseudoRandomNumberGenerator(seed=seed_2, sample_size=max_iterations).generate() - 1
    i = 0
    while i<max_iterations:
        s = w_1[i]**2 + w_2[i]**2
        if s<1:
            t = np.sqrt(-2*np.log(s)/s)
            return (w_1[i]*t, w_2[i]*t)
        i = i+1



def ziggurat_algorithm(x_guess: np.ndarray, sample_size: int=10_000, max_iterations: int=1_000, seed_1: int=78_321,
                       seed_2: int=32_456) -> np.ndarray:
    type_check(value=x_guess, type_=np.ndarray, value_name="x")
    """
    ## Description
    The Ziggurat algorithm is a fast method for generating random samples from a normal distribution.
    ### Input:
        - x_guess (np.ndarray): Initial guess values
        - sample_size (int, optional): Number of random samples to generate
        - max_iterations (int, optional): Maximum number of iterations for the algorithm
        - seed_1 (int, optional): Seed for the first pseudo-random number generator
        - seed_2 (int, optional): Seed for the second pseudo-random number generator
    ### Output:
        - Sample arrays (np.ndarray) generated from the normal distribution
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Ziggurat_algorithm
        - Original Source: https://doi.org/10.1145/1464291.1464310
    """
    sample_size = int(sample_size)
    max_iterations = int(max_iterations)
    seed_1 = int(seed_1)
    seed_2 = int(seed_2)
    normal_dist = NormalDistribution(mu=0, sigma=1)
    Y = normal_dist.pdf(x=x_guess)
    Z = np.ones(sample_size)
    for j in range(sample_size):
        U_1 = 2*FibonacciPseudoRandomNumberGenerator(seed=seed_1, sample_size=len(x_guess)).generate() - 1
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=seed_2, sample_size=len(x_guess)).generate()
        i = 0
        while (Z[j]==1) and i<max_iterations:
            i = np.random.randint(0, len(x_guess))
            x = U_1[i]*x_guess[i]
            if abs(x)<x_guess[i-1]:
                Z[j] = x
            else:
                y = Y[i] + U_2[i]*(Y[i-1]-Y[i])
                point = normal_dist.pdf(x=x)
                if y<point:
                    Z[j] = x
            i = i+1
        seed_1 = int(np.abs(np.ceil(Z[j]*714_025)))
        seed_2 = int(np.abs(np.ceil(Z[j]*714_025)))
    return Z