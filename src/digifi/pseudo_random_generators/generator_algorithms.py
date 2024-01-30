from typing import Callable
import numpy as np
from src.digifi.utilities.general_utils import (compare_array_len, type_check)
from src.digifi.probability_distributions.continuous_probability_distributions import NormalDistribution
from src.digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator



def accept_reject_method(f_x: Callable, g_x: Callable, Y_sample: np.ndarray, M: float, uniform_sample: np.ndarray,
                         sample_size: int=10_000) -> np.ndarray:
    compare_array_len(array_1=Y_sample, array_2=uniform_sample, array_1_name="Y_sample", array_2_name="uniform_sample")
    g = g_x(Y_sample)
    f = f_x(Y_sample)
    X_sample = np.array([])
    for j in range(sample_size):
        if (uniform_sample[j]*M*g[j])<=f[j]:
            X_sample = np.append(X_sample, Y_sample[j])
    return X_sample



def inverse_transform_method(lambda_param: float, sample_size: int, seed: int) -> np.ndarray: #exponential
    uniform_samples = FibonacciPseudoRandomNumberGenerator(seed=seed, sample_size=sample_size).generate()
    return -np.log(1 - uniform_samples) / lambda_param



def box_muller_algorithm(uniform_array_1: np.ndarray, uniform_array_2: np.ndarray) -> (np.ndarray, np.ndarray):
    return (np.sqrt(-2*np.log(uniform_array_1))*np.cos(2*np.pi*uniform_array_2),
            np.sqrt(-2*np.log(uniform_array_1))*np.sin(2*np.pi*uniform_array_2))



def marsaglia_method(max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> (float, float):
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