import numpy as np
from src.digifi.pseudo_random_generators.general import PseudoRandomGeneratorInterface
from src.digifi.pseudo_random_generators.generator_algorithms import (accept_reject_method, inverse_transform_method,
                                                                      box_muller_algorithm, marsaglia_method, ziggurat_algorithm)
from src.digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator
from src.digifi.probability_distributions.continuous_probability_distributions import (NormalDistribution, LaplaceDistribution)



class StandardNormalAcceptRejectPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for standard normal distribution.
    It samples the Laplace distribution to generate the standard normal distribution (i.e., exponential tilting).

    ### Input:
    - sample_size (int): The size of the sample to generate.
    - lap_p (float): Probability parameter for the Laplace distribution.
    - seed_1 (int): Seed for the first random number generator.
    - seed_2 (int): Seed for the second random number generator.

    ### Output:
        - Array of pseudo-random numbers following a standard normal distribution.

    ### Formula:
        - Accept if \( U \cdot M \cdot g(Y) \leq f(Y) \), where \( f \) is the target PDF (normal), \( g \) is the proposal PDF (Laplace), \( U \) is uniform random, \( Y \) is from \( g \), and \( M \) is a constant.

    ### Links:
        - https://en.wikipedia.org/wiki/Rejection_sampling
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
    ## Description
    Pseudo-random number generator for standard normal distribution.
    It returns the array of values from sampling an inverse CDF.

    ### Input:
        - sample_size (int): Number of random samples to generate.
        - seed (int): Seed for the pseudo-random number generator.

    ### Output:
        - Array of pseudo-random numbers following a standard normal distribution.

    ### Formula:
        - \( X = F^{-1}(U) \), where \( F^{-1} \) is the inverse CDF of the normal distribution and \( U \) is a uniform random variable.

    ### Links:
        - https://en.wikipedia.org/wiki/Inverse_transform_sampling
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
    ## Description
    Pseudo-random number generator for standard normal distribution.
    It returns two independent pseudo-random arrays.

    ### Input:
    - sample_size (int): Number of random samples to generate.
    - seed_1 (int): Seed for the first uniform random number generator.
    - seed_2 (int): Seed for the second uniform random number generator.

    ### Output:
        - Two arrays of pseudo-random numbers following a standard normal distribution.

    ### Formula:
        - \( Z_0 = \sqrt{-2 \ln U_1} \cos(2\pi U_2) \)
        - \( Z_1 = \sqrt{-2 \ln U_1} \sin(2\pi U_2) \)
        - \( U_1, U_2 \) are independent uniform random variables.

    ### Links:
        - https://en.wikipedia.org/wiki/Box–Muller_transform
    """
    def __init__(self, sample_size: int=10_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Two independent arrays of pseudo-random generated numbers based on the Box-Muller Algorithm.
        """
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).generate()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).generate()
        # Box-Muller algorithm
        return box_muller_algorithm(uniform_array_1=U_1, uniform_array_2=U_2)



class StandardNormalMarsagliaMethodPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for standard normal distribution.
    It returns two independent pseudo-random arrays.

    ### Input:
    - sample_size (int): Number of random samples to generate.
    - max_iterations (int): Maximum number of iterations for the algorithm.
    - seed_1 (int): Seed for the first uniform random number generator.
    - seed_2 (int): Seed for the second uniform random number generator.

    ### Output:
        - Two arrays of pseudo-random numbers following a standard normal distribution.

    ### Formula:
        - \( S = W_1^2 + W_2^2 \)
        - If \( S < 1 \), then \( Z_0 = W_1 \sqrt{-2 \ln S / S} \), \( Z_1 = W_2 \sqrt{-2 \ln S / S} \)

    ### Links:
        - https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    def __init__(self, sample_size: int=10_000, max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.max_iterations = int(max_iterations)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
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
    ## Description
    Pseudo-random number generator for standard normal distribution.

    ### Input:
    - sample_size (int): Number of random samples to generate.
    - regions (int): Number of regions for the Ziggurat method.
    - max_iterations (int): Maximum number of iterations for the algorithm.
    - seed_1 (int): Seed for the first uniform random number generator.
    - seed_2 (int): Seed for the second uniform random number generator.

    ### Output:
        - Array of pseudo-random numbers following a standardnormal distribution.

    ### Input:
        - dx (float, optional): The step size for the initial guess values. Default is 0.001.
        - limit (int, optional): The limit for the x-axis in the normal distribution. Default is 6.

    ### Output:
        - Array of pseudo-random numbers following a standard normal distribution.

    ### Formula:
        - The Ziggurat algorithm divides the area under the PDF into several layers and segments for efficient sampling.
        - Specific mathematical description of the algorithm involves conditional sampling and rejection based on the computed area.

    ### Links:
        - https://en.wikipedia.org/wiki/Ziggurat_algorithm
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