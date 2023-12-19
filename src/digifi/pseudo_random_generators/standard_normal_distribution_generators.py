from typing import Union
import numpy as np
import plotly.graph_objects as go
from src.digifi.pseudo_random_generators.general import PseudoRandomGeneratorInterface
from src.digifi.pseudo_random_generators.generator_algorithms import (accept_reject_method, box_muller_algorithm, marsaglia_method,
                                                                      ziggurat_algorithm)
from src.digifi.pseudo_random_generators.uniform_distribution_generators import FibonacciPseudoRandomNumberGenerator
from src.digifi.plots.pseudo_random_generators_plots import (plot_pdf, plot_2d_scattered_points, plot_3d_scattered_points)
from src.digifi.probability_distributions.continuous_probability_distributions import (NormalDistribution, LaplaceDistribution)
# TODO: Add inverse transformation method



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
    
    def get_randomized_array(self):
        """
        Array of pseudo-random generated numbers based on the Accept-Reject Method and the probability of the Laplace Distribution lap_p.
        """
        M = np.sqrt(2*np.exp(1)/np.pi)
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).get_randomized_array()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).get_randomized_array()
        L = np.ones(self.sample_size)
        N = []
        # Laplace distribution sampling
        laplace_dist = LaplaceDistribution(mu=0, b=1)
        # TODO: Replace sampling with the inverse tranform method based on inverse CDF of a Laplace distribution
        for i in range(0, self.sample_size):
            if U_1[i]<(1-self.lap_p):
                L[i] = np.log(2*U_1[i])
            else:
                L[i] = -np.log(2*(1-U_1[i]))
        # Accept-reject algorithm
        standard_normal_dist = NormalDistribution(mu=0, sigma=1)
        return accept_reject_method(f_x=standard_normal_dist.pdf, g_x=laplace_dist.pdf, Y_sample=L, M=M, uniform_sample=U_2,
                                    sample_size=self.sample_size)

    
    def plot_pdf(self, n_bins: int=100, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Histogram plot of the probability density function of an array generated with standard normal accept-reject generator.
        """
        return plot_pdf(points=self.get_randomized_array(), n_bins=n_bins, return_fig_object=return_fig_object)
    
    def plot_2d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        2D scatter plot of the pseudo-random points generated with standard normal accept-reject generator.
        """
        return plot_2d_scattered_points(points=self.get_randomized_array(), return_fig_object=return_fig_object)
    
    def plot_3d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        3D scatter plot of the pseudo-random points generated with standard normal accept-reject generator.
        """
        return plot_3d_scattered_points(points=self.get_randomized_array(), return_fig_object=return_fig_object)



class StandardNormalBoxMullerAlgorithmPseudoRandomNumberGenerator(PseudoRandomGeneratorInterface):
    """
    Pseudo-random number generator for standard normal distribution.
    It returns two independent pseudo-random arrays.
    """
    def __init__(self, sample_size: int=10_000, seed_1: int=78_321, seed_2: int=32_456) -> None:
        self.sample_size = int(sample_size)
        self.seed_1 = int(seed_1)
        self.seed_2 = int(seed_2)
    
    def get_randomized_array(self) -> (np.ndarray, np.ndarray):
        """
        Two independent arrays of pseudo-random generated numbers based on the Box-Muller Algorithm.
        """
        U_1 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_1, sample_size=self.sample_size).get_randomized_array()
        U_2 = FibonacciPseudoRandomNumberGenerator(seed=self.seed_2, sample_size=self.sample_size).get_randomized_array()
        # Box-Muller algorithm
        return box_muller_algorithm(uniform_array_1=U_1, uniform_array_2=U_2)
    
    def plot_pdf(self, n_bins: int=100, display_both: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Histogram plot of the probability density function of an array generated with Box-Muller algorithm.
        """
        points = self.get_randomized_array()
        fig = plot_pdf(points=points[0], n_bins=n_bins, return_fig_object=True)
        if bool(display_both):
            fig.add_trace(go.Histogram(x=points[1], histnorm="probability density", nbinsx=int(n_bins), name="Probabilityu Density Function 2"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None
    
    def plot_2d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        2D scatter plot of the pseudo-random points generated withl Box-Muller algorithm.
        """
        points = self.get_randomized_array()
        x = points[0]
        y = points[1]
        fig = go.Figure(go.Scatter(x=x, y=y, name="2D Scatter Plot", mode="markers"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None
    
    def plot_3d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        3D scatter plot of the pseudo-random points generated with Box-Muller algorithm.
        """
        points = self.get_randomized_array()
        x = points[0][0:len(points[0])-1]
        y = points[1]
        z = points[0][1: len(points[0])]
        fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, name="3D Scatter Plot", mode="markers"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None



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
    
    def get_randomized_array(self) -> (np.ndarray, np.ndarray):
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
    
    def plot_pdf(self, n_bins: int=100, display_both: bool=False, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Histogram plot of the probability density function of an array generated with Marsaglia method.
        """
        points = self.get_randomized_array()
        fig = plot_pdf(points=points[0], n_bins=n_bins, return_fig_object=True)
        if bool(display_both):
            fig.add_trace(go.Histogram(x=points[1], histnorm="probability density", nbinsx=int(n_bins), name="Probability Density Function 2"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None
    
    def plot_2d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        2D scatter plot of the pseudo-random points generated with Marsaglia method.
        """
        points = self.get_randomized_array()
        x = points[0]
        y = points[1]
        fig = go.Figure(go.Scatter(x=x, y=y, name="2D Scatter Plot", mode="markers"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None
    
    def plot_3d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        3D scatter plot of the pseudo-random points generated with Marsaglia method.
        """
        points = self.get_randomized_array()
        x = points[0][0:len(points[0])-1]
        y = points[1]
        z = points[0][1: len(points[0])]
        fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, name="3D Scatter Plot", mode="markers"))
        if bool(return_fig_object):
            return fig
        else:
            fig.show()
            return None



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
    
    def get_randomized_array(self, dx: float=0.001, limit: int=6) -> np.ndarray:
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
    
    def plot_pdf(self, n_bins: int=100, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        Histogram plot of the probability density function of an array generated with standard normal ziggurat algorithm.
        """
        return plot_pdf(points=self.get_randomized_array(), n_bins=n_bins, return_fig_object=return_fig_object)
    
    def plot_2d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        2D scatter plot of the pseudo-random points generated with standard normal ziggurat algorithm.
        """
        return plot_2d_scattered_points(points=self.get_randomized_array(), return_fig_object=return_fig_object)
    
    def plot_3d_scattered_points(self, return_fig_object: bool=False) -> Union[go.Figure, None]:
        """
        3D scatter plot of the pseudo-random points generated with standard normal ziggurat algorithm.
        """
        return plot_3d_scattered_points(points=self.get_randomized_array(), return_fig_object=return_fig_object)