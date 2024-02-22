import numpy as np
from digifi.pseudo_random_generators.general import PseudoRandomGeneratorInterface



class LinearCongruentialPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for uniform distribution.
    ### Input:
        - seed (int): Seed of the generator
        - sample_size (int): Number of pseudo-random numbers to generate
        - M (int): Mod of the linear congruential generator
        - a (int): Multiplierof the linear congruential generator
        - b (int): Increment of the linear congruential generator
    ### LaTeX Formula:
        - N_{i} = (aN_{i-1}+b) mod M
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Linear_congruential_generator
        - Original Source: https://archive.org/details/proceedings_of_a_second_symposium_on_large-scale_/mode/2up
    """
    def __init__(self, seed: int=12_345, sample_size: int=10_000, M: int=244_944, a: int=1_597, b: int=51_749) -> None:
        if seed<0:
            raise ValueError("The seed must be a positive integer")
        if sample_size<=0:
            raise ValueError("The sample must be a positive integer")
        self.seed = int(seed)
        self.sample_size = int(sample_size)
        self.M = int(M)
        self.a = int(a)
        self.b = int(b)

    def generate(self) -> np.ndarray:
        """
        ## Description
        Array of pseudo-random generated numbers based on Linear Congruential Generator.
        ### Output:
            - An array (np.ndarray) pseudo-random numberss following Uniform distribution
        """
        u = np.zeros(self.sample_size)
        u[0] = self.seed
        for i in range(1, self.sample_size):
            u[i] = (self.a*u[i-1] + self.b)%self.M
        return u/self.M



class FibonacciPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    ## Description
    Pseudo-random number generator for uniform distribution.
    ### Input:
        - mu (int): First primitive polynomial degree
        - nu (int): Second primitive polynomial degree
        - seed (int): Seed of the generator
        - sample_size (int): Number of pseudo-random numbers to generate
        - M (int): Mod of the linear congruential generator
        - a (int): Multiplierof the linear congruential generator
        - b (int): Increment of the linear congruential generator
    ### LaTeX Formula:
        - N_{i} = (N_{i-nu}-N_{i-mu}) mod M
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Lagged_Fibonacci_generator
        - Original Source: N/A
    """
    def __init__(self, mu: int=5, nu: int=17, seed: int=12_345, sample_size: int=10_000, M: int=714_025, a: int=1_366, b: int=150_889) -> None:
        if seed<0:
            raise ValueError("The seed must be a positive integer")
        if sample_size<=0:
            raise ValueError("The sample must be a positive integer")
        self.mu = int(mu)
        self.nu = int(nu)
        self.seed = int(seed)
        self.sample_size = int(sample_size)
        self.M = int(M)
        self.a = int(a)
        self.b = int(b)
    
    def generate(self) -> np.ndarray:
        """
        ## Description
        Array of pseudo-random generated numbers based on Fibonacci Generator.
        ### Output:
            - An array (np.ndarray) pseudo-random numberss following Uniform distribution
        """
        u = np.zeros(self.sample_size)
        u[0] = self.seed
        for i in range(1, self.sample_size):
            u[i] = (self.a*u[i-1] + self.b)%self.M
        for i in range(self.nu+1, self.sample_size):
            u[i] = (u[i-self.nu]-u[i-self.mu])%self.M
        return u/self.M