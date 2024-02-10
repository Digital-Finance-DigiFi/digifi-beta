import numpy as np
from src.digifi.pseudo_random_generators.general import PseudoRandomGeneratorInterface
from src.digifi.pseudo_random_generators.generator_algorithms import (accept_reject_method, inverse_transform_method,
                                                                      box_muller_algorithm, marsaglia_method, ziggurat_algorithm)
from src.digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator
from src.digifi.probability_distributions.continuous_probability_distributions import (NormalDistribution, LaplaceDistribution)



class StandardNormalAcceptRejectPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    It samples the Laplace distribution to generate the standard normal distribution (i.e., exponential tilting).
    """
    def __init__(self, sample_size: int=10_000, lap_p: float=0.5, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.lap_p = float(lap_p)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self):
        """
        Array of pseudo-random generated numbers based on the Accept-Reject Method and the probability of the Laplace Distribution lap_p.
        """
        M = np.sqrt(2*np.exp(1)/np.pi)
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).generate()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).generate()
        # Laplace distribution sampling
        laplace_dist = LaplaceDistribution(mu=0, b=1)
        L = laplace_dist.inverse_cdf(p=U_1)
        # Accept-reject algorithm
        standard_normal_dist = NormalDistribution(mu=0, sigma=1)
        return accept_reject_method(f_x=standard_normal_dist.pdf, g_x=laplace_dist.pdf, Y_sample=L, M=M, uniform_sample=U_2,
                                    sample_size=self.sample_size)



class StandardNormalInverseTransformMethodPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    It returns the array of values from sampling an inverse CDF.
    """
    def __init__(self, sample_size: int=10_000, seed: int=78_321) -> None:
        self.sample_size = int(sample_size)
        self.seed = int(seed)
    
    def generate(self) -> np.ndarray:
        """
        Array of pseudo-random generated numbers based on the Inverse Transform Method.
        """
        normal_dist = NormalDistribution(mu=0.0, sigma=1.0)
        return inverse_transform_method(f_x=normal_dist.inverse_cdf, sample_size=self.sample_size, seed=self.seed)



class StandardNormalBoxMullerAlgorithmPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    It returns two independent pseudo-random arrays.
    """
    def __init__(self, sample_size: int=10_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> (np.ndarray, np.ndarray):
        """
        Two independent arrays of pseudo-random generated numbers based on the Box-Muller Algorithm.
        """
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).generate()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).generate()
        # Box-Muller algorithm
        return box_muller_algorithm(uniform_array_1=U_1, uniform_array_2=U_2)



class StandardNormalMarsagliaMethodPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    It returns two independent pseudo-random arrays.
    """
    def __init__(self, sample_size: int=10_000, max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.max_iterations = int(max_iterations)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> (np.ndarray, np.ndarray):
        """
        Two independent arrays of pseudo-random generated numbers based on the Marsaglie Method.
        """
        Z_1 = np.ones(self.sample_size)
        Z_2 = np.ones(self.sample_size)
        # Marsaglia method
        for i in range(0, self.sample_size):
            Z_1[i], Z_2[i] = marsaglia_method(max_iterations=self.max_iterations, seed_1=self.seed_1, seed_2=self.seed_2)
            self.seed_1 = int(np.abs(np.ceil(Z_1[i]*714_025)))
            self.seed_2 = int(np.abs(np.ceil(Z_2[i]*714_025)))
        return (Z_1, Z_2)



class StandardNormalZigguratAlgorithmPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    """
    def __init__(self, sample_size: int=10_000, regions: int=256, max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.rectangle_size = 1/int(regions)
        self.max_iterations = int(max_iterations)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self, dx: float=0.001, limit: int=6) -> np.ndarray:
        """
        Array of pseudo-random generated numbers based on the Ziggurat Algorithm.
        """
        x_guess = np.array([])
        current_x = 0
        current_area, rectangle_length = 0, 0
        normal_dist = NormalDistribution(mu=0, sigma=1)
        # Initial guess
        while current_x < limit:
            rectangle_length = rectangle_length+dx
            current_area = (normal_dist.pdf(x=current_x)-normal_dist.pdf(x=rectangle_length))*rectangle_length
            if current_area > self.rectangle_size:
                x_guess = np.append(x_guess, rectangle_length)
                current_x = rectangle_length
        # Ziggurat algorithm
        return ziggurat_algorithm(x_guess=x_guess, sample_size=self.sample_size, max_iterations=self.max_iterations, seed_1=self.seed_1,
                                  seed_2=self.seed_2)