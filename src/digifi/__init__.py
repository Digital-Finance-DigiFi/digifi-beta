# TODO: Add references of algorithms to doctrings (Wikipedia reference is applicable, Original Source reference if applicable)

# Utilities

from .utilities.maths_utils import (factorial, n_choose_r)

from .utilities.time_value_utils import (CompoundingType, Cashflow, present_value, net_present_value, future_value, internal_rate_of_return,
                                         ptp_compounding_transformation, ptc_compounding_transformation, ctp_compounding_transformation,
                                         real_interest_rate, forward_interest_rate, Compounding, Perpetuity, Annuity)

from .utilities.general_utils import (verify_array, compare_array_len, rolling)

from .utilities.bonds_utils import (bond_price_from_yield, bootstrap)

from .utilities.corporate_finance_utils import (dol, pe_ratio, dividend_yield, book_value, pb_ratio, cost_of_equity_capital,
                                                maximum_drawdown)

# Lattice-based models

from .lattice_based_models.general import (LatticeModelPayoffType, LatticeModelInterface)

from .lattice_based_models.binomial_models import (binomial_tree_nodes, binomial_model, BrownianMotionBinomialModel)

from .lattice_based_models.trinomial_models import (trinomial_tree_nodes, trinomial_model, BrownianMotionTrinomialModel)

# Corporate finance

from .corporate_finance.linear_regression_models import (CAPMType, CAPMSolutionType, CAPM)

# Financial instruments

from .financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                            FinancialInstrumentAssetClass)

from .financial_instruments.bonds import (BondType, BondStruct, BondInterface, YtMMethod, Bond)

from .financial_instruments.stocks import (StockStruct, StockInteraface, Stock)

from .financial_instruments.derivatives import (ContractType, OptionType, OptionPayoffType, OptionPricingMethod, FuturesContractStruct,
                                                OptionStruct, FuturesContractInterface, OptionInterface, FuturesContract, Option)

from .financial_instruments.rates_and_swaps import (ForwardRateAgreementStruct, ForwardRateAgreement)

# Stochastic models

from .stochastic_processes.general import (StochasticProcessInterface, StochasticProcess)

from .stochastic_processes.standard_stochastic_models import (FellerSquareRootProcessMethod, ArithmeticBrownianMotion,
                                                           GeometricBrownianMotion,OrnsteinUhlenbeckProcess, BrownianBridge,
                                                           FellerSquareRootProcess)

from .stochastic_processes.stochastic_volatility_models import (ConstantElasticityOfVariance, HestonStochasticVolatility,
                                                             VarianceGammaProcess)

from .stochastic_processes.jump_diffusion_models import (MertonJumpDiffusionProcess, KouJumpDiffusionProcess)

# Technical indicators

from .technical_indicators.technical_indicators import (ema, sma, macd, bollinger_bands, rsi, adx, obv)

# Probability distributions

from .probability_distributions.general import (ProbabilityDistributionType, ProbabilityDistributionStruct,
                                                ProbabilityDistributionInterface, skewness, kurtosis)

from .probability_distributions.discrete_probability_distributions import (BernoulliDistribution, BinomialDistribution,
                                                                           DiscreteUniformDistribution, PoissonDistribution)

from .probability_distributions.continuous_probability_distributions import (ContinuousUniformDistribution, NormalDistribution,
                                                                             ExponentialDistribution, LaplaceDistribution)

# Pseudo-random generators

from .pseudo_random_generators.general import (PseudoRandomGeneratorInterface)

from .pseudo_random_generators.generator_algorithms import (accept_reject_method, inverse_transformation_method, box_muller_algorithm,
                                                            marsaglia_method, ziggurat_algorithm)

from .pseudo_random_generators.uniform_distribution_generators import (LinearCongruentialPseudoRandomNumberGenerator,
                                                                       FibonacciPseudoRandomNumberGenerator)

from .pseudo_random_generators.standard_normal_distribution_generators import (StandardNormalAcceptRejectPseudoRandomNumberGenerator,
                                                                               StandardNormalBoxMullerAlgorithmPseudoRandomNumberGenerator,
                                                                               StandardNormalMarsagliaMethodPseudoRandomNumberGenerator,
                                                                               StandardNormalZigguratAlgorithmPseudoRandomNumberGenerator)

# Portfolio applications

from .portfolio_applications.general import (ReturnsMethod, ArrayRetrunsType, PortfolioInstrumentStruct, prices_to_returns,
                                             returns_average, returns_std, returns_variance, PortfolioInterface)

from .portfolio_applications.portfolio_management import (Portfolio, InstrumentsPortfolio)

# Market making

from .market_making.order_book import (volume_imbalance)

# Plots

from .plots.general_plots import (plot_candlestick_chart)

from .plots.stochastic_models_plots import (plot_stochastic_paths)

from .plots.technical_indicators_plots import (plot_sma, plot_ema, plot_macd, plot_bollinger_bands, plot_rsi, plot_adx)

from .plots.pseudo_random_generators_plots import (plot_pdf, plot_2d_scattered_points, plot_3d_scattered_points)

from .plots.financial_instuments_plots import (plot_option_payoff, plot_option_value_surface)

from .plots.portfolio_application_plots import (plot_portfolio_cumulative_returns, plot_efficient_frontier)