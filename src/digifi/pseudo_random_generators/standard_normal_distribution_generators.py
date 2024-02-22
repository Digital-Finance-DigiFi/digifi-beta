import numpy as np
from digifi.pseudo_random_generators.general import PseudoRandomGeneratorInterface
from digifi.pseudo_random_generators.generator_algorithms import (accept_reject_method, inverse_transform_method,
                                                                      box_muller_algorithm, marsaglia_method, ziggurat_algorithm)
from digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator
from digifi.probability_distributions.continuous_probability_distributions import (NormalDistribution, LaplaceDistribution)



class StandardNormalAcceptRejectPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for standard normal distribution.\n
    It samples the Laplace distribution to generate the standard normal distribution (i.e., exponential tilting).
    ### Input:
        - sample_size (int): The size of the sample to generate
        - lap_p (float): Probability parameter for the Laplace distribution
        - seed_1 (int): Seed for the first random number generator
        - seed_2 (int): Seed for the second random number generator
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Rejection_sampling
        - Original Source: N/A
    """
    def __init__(self, sample_size: int=10_000, lap_p: float=0.5, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.lap_p = float(lap_p)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self):
        """
        ## Description
        Array of pseudo-random generated numbers based on the Accept-Reject Method and the probability of the Laplace Distribution lap_p.
        ### Output:
            - An array (np.ndarray) of pseudo-random numbers following a standard normal distribution
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
    Pseudo-random number generator for standard normal distribution.\n
    It returns the array of values from sampling an inverse CDF.
    ### Input:
        - sample_size (int): Number of random samples to generate
        - seed (int): Seed for the pseudo-random number generator
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Inverse_transform_sampling
        - Original Source: N/A
    """
    def __init__(self, sample_size: int=10_000, seed: int=78_321) -> None:
        self.sample_size = int(sample_size)
        self.seed = int(seed)
    
    def generate(self) -> np.ndarray:
        """
        ## Description
        Array of pseudo-random generated numbers based on the Inverse Transform Method.
        ### Output:
            - An array (np.ndarray) of pseudo-random numbers following a standard normal distribution
        """
        normal_dist = NormalDistribution(mu=0.0, sigma=1.0)
        return inverse_transform_method(f_x=normal_dist.inverse_cdf, sample_size=self.sample_size, seed=self.seed)



class StandardNormalBoxMullerAlgorithmPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for standard normal distribution.\n
    It returns two independent pseudo-random arrays.
    ### Input:
        - sample_size (int): Number of random samples to generate
        - seed_1 (int): Seed for the first uniform random number generator
        - seed_2 (int): Seed for the second uniform random number generator
    ### LaTeX Formula:
        - Z_{0} = \\sqrt{-2ln(U_{1})} \\cos(2\\pi U_{2})
        - Z_{1} = \\sqrt{-2ln(U_{1})} \\sin(2\\pi U_{2})
        - U_{1}, U_{2} are independent uniform random variables
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Box–Muller_transform
        - Original Source: https://doi.org/10.1214%2Faoms%2F1177706645
    """
    def __init__(self, sample_size: int=10_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ## Description
        Two independent arrays of pseudo-random generated numbers based on the Box-Muller Algorithm.
        ### Output:
            - Two arrays (np.ndarray) of pseudo-random numbers following a standard normal distribution
        """
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).generate()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).generate()
        # Box-Muller algorithm
        return box_muller_algorithm(uniform_array_1=U_1, uniform_array_2=U_2)



class StandardNormalMarsagliaMethodPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for standard normal distribution.\n
    It returns two independent pseudo-random arrays.
    ### Input:
        - sample_size (int): Number of random samples to generate
        - max_iterations (int): Maximum number of iterations for the algorithm
        - seed_1 (int): Seed for the first uniform random number generator
        - seed_2 (int): Seed for the second uniform random number generator
    ### LaTeX Formula:
        - S = W^{2}_{1} + W^{2}_{2}
        - If S < 1, then:
        - Z_{0} = W_{1} \\sqrt{\\frac{-2ln(S)}{S}}
        - Z_{1} = W_{2} \\sqrt{\\frac{-2ln(S)}{S}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Marsaglia_polar_method
        - Original Source: https://doi.org/10.1137%2F1006063
    """
    def __init__(self, sample_size: int=10_000, max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.max_iterations = int(max_iterations)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ## Description
        Two independent arrays of pseudo-random generated numbers based on the Marsaglie Method.
        ### Output:
            - Two arrays (np.ndarray) of pseudo-random numbers following a standard normal distribution
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
        - sample_size (int): Number of random samples to generate
        - regions (int): Number of regions for the Ziggurat method
        - max_iterations (int): Maximum number of iterations for the algorithm
        - seed_1 (int): Seed for the first uniform random number generator
        - seed_2 (int): Seed for the second uniform random number generator
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Ziggurat_algorithm
        - Original Source: https://doi.org/10.1145/1464291.1464310
    """
    def __init__(self, sample_size: int=10_000, regions: int=256, max_iterations: int=1_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.rectangle_size = 1/int(regions)
        self.max_iterations = int(max_iterations)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def generate(self, dx: float=0.001, limit: int=6) -> np.ndarray:
        """
        ## Description
        Array of pseudo-random generated numbers based on the Ziggurat Algorithm.
        ### Input:
            - dx (float, optional): The step size for the initial guess values
            - limit (int, optional): The limit for the x-axis in the normal distribution
        ### Output:
            - An array (np.ndarray) of pseudo-random numbers following a standard normal distribution
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