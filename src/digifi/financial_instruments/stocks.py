from typing import (Union, Any)
import abc
from dataclasses import dataclass
from enum import Enum
import numpy as np
import scipy
from digifi.utilities.general_utils import (compare_array_len, type_check, DataClassValidation)
from digifi.utilities.time_value_utils import (CompoundingType, Compounding, Perpetuity)
from digifi.financial_instruments.general import (FinancialInstrumentStruct, FinancialInstrumentInterface, FinancialInstrumentType,
                                                      FinancialInstrumentAssetClass)
from digifi.portfolio_applications.general import PortfolioInstrumentStruct



class QuoteValues(Enum):
    PER_SHARE = 1
    TOTAL = 2



class StockValuationType(Enum):
    DIVIDEND_DISCOUNT_MODEL = 1
    VALUATION_BY_COMPARABLES = 2



@dataclass(slots=True)
class ValuationByComparablesParams(DataClassValidation):
    """
    ## Description
    Parameters for valuation by comparables.
    ### Input:
        - valuations (np.ndarray): Array of valuations of the companies
        - pe_ratios (Union[np.ndarray, None]): Array of P/E ratios of the companies
        - pb_ratios (Union[np.ndarray, None]): Array of P/B ratios of the companies
        - ev_to_ebitda (Union[np.ndarray, None]): Array of EV/EBITDA ratios of the companies
    """
    valuations: np.ndarray
    pe_ratios: Union[np.ndarray, None]
    pb_ratios: Union[np.ndarray, None]
    ev_to_ebitda: Union[np.ndarray, None]

    def __set_length(self, feature_name: str) -> None:
        """
        ## Description
        Ensures that if the feature is not an array - an array of zeros is used,
        or if the feature is an array - checks that the length of feature is the same as the valuations array.
        """
        feature_name = str(feature_name)
        feature_array: np.ndarray = getattr(self, feature_name)
        if isinstance(feature_array, type(None)):
            setattr(self, feature_name, np.zeros(len(self.valuations)))
        else:
            compare_array_len(array_1=self.valuations, array_2=feature_array, array_1_name="valuations", array_2_name=feature_name)

    def __post_init__(self) -> None:
        super(ValuationByComparablesParams, self).__post_init__()
        N_DATAPOINTS = 5
        if len(self.valuations)==0:
            raise ValueError("No valuations provided.")
        if len(self.valuations)<N_DATAPOINTS:
            raise ValueError("Too few data points provided. Valuation by comparables requires at least {} companies to be included for valuation.".format(N_DATAPOINTS))
        # P/E ratio validation
        self.__set_length(feature_name="pe_ratios")
        # P/B ratio validation
        self.__set_length(feature_name="pb_ratios")
        # EV/EBITDA validation
        self.__set_length(feature_name="ev_to_ebitda")
        



@dataclass(slots=True)
class StockStruct(DataClassValidation):
    """
    ## Description
    Parameters for the Stock class.
    ### Input:
        - price_per_share (float): Price per share
        - n_shares_outsdtanding (int): Number of shares outstanding
        - dividend_per_share (float): Dividend per share
        - earnings_per_share (float): Earnings per share (EPS)
        - quote_values (QuoteValues): Determines how output of Stock classs methods will be quoted (i.e., PER_SHARE - for values per share, TOTAL - for total values)
        - initial_price (float): Initial price at which the stock is purchased
        - compounding_type (CompoundingType): Compounding type used to discount cashflows
        - dividend_growth_rate (float): Growth rate of the dividend payouts
        - dividend_compounding_frequency (float): Compounding frequency of the dividend payouts
    """
    price_per_share: float
    n_shares_outstanding: int
    dividend_per_share: float
    earnings_per_share: float
    quote_values: QuoteValues
    initial_price: float = 0.0
    compounding_type: CompoundingType = CompoundingType.PERIODIC
    dividend_growth_rate: float = 0.0
    dividend_compounding_frequency: float = 1.0



class StockInteraface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, "cost_of_equity_capital") and
                callable(subclass.cost_of_equity_capital) and
                hasattr(subclass, "dividend_discount_model") and
                callable(subclass.dividend_discount_model) and
                hasattr(subclass, "payout_ratio") and
                callable(subclass.payout_ratio) and
                hasattr(subclass, "plowback_ratio") and
                callable(subclass.plowback_ratio) and
                hasattr(subclass, "return_on_equity") and
                callable(subclass.return_on_equity) and
                hasattr(subclass, "dividend_growth_rate") and
                callable(subclass.dividend_growth_rate) and
                hasattr(subclass, "present_value_of_growth_opportunities") and
                callable(subclass.present_value_of_growth_opportunities) and
                hasattr(subclass, "valuation") and
                callable(subclass.valuation))
    
    @abc.abstractmethod
    def cost_of_equity_capital(self) -> float:
        """
        ## Description
        Calculate cost of equity capital.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def dividend_discount_model(self) -> float:
        """
        ## Description
        Create dividend discount model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def valuation_by_comparables(self) -> float:
        """
        ## Description
        Valuate stock.
        """
        # Valuation by comparables - P/E, Market-to-Book Ratio, etc.; discounted cash flow - post-horizon PVGO
        raise NotImplementedError
    
    @abc.abstractmethod
    def dividend_growth_rate(self) -> float:
        """
        ## Description
        Calculate dividend growth rate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def present_value_of_growth_opportunities(self) -> float:
        """
        ## Description
        Calculate present value of growth opportunities.
        """
        raise NotImplementedError



class Stock(FinancialInstrumentInterface, StockInteraface):
    """
    ## Description
    Stock financial instrument and its methods.
    ### Input:
        - stock_struct (StockStruct): Parameters for defining a Stock instance
        - financial_instrument_struct (FinancialInstrumentStruct): Parameters for defining regulatory categorization of an instrument
        - portfolio_instrument_struct (PortfolioInstrumentStruct): Parameters for defining historical data for portfolio construction and applications
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Stock
        - Original Source: N/A
    """
    def __init__(self, stock_struct: StockStruct,
                 financial_instrument_struct: FinancialInstrumentStruct=FinancialInstrumentStruct(instrument_type=FinancialInstrumentType.CASH_INSTRUMENT,
                                                                                                  asset_class=FinancialInstrumentAssetClass.EQUITY_BASED_INSTRUMENT,
                                                                                                  identifier="0"),
                 portfolio_instrument_struct: PortfolioInstrumentStruct=PortfolioInstrumentStruct(portfolio_price_array=np.array([]),
                                                                                                  portfolio_time_array=np.array([]),
                                                                                                  portfolio_predicatble_income=np.array([]))) -> None:
        # Arguments validation
        type_check(value=stock_struct, type_=StockStruct, value_name="stock_struct")
        type_check(value=financial_instrument_struct, type_=FinancialInstrumentStruct, value_name="financial_instrument_struct")
        type_check(value=portfolio_instrument_struct, type_=PortfolioInstrumentStruct, value_name="portfolio_instrument_struct")
        # StockStruct parameters
        self.price_per_share = stock_struct.price_per_share
        self.n_shares_outstanding = stock_struct.n_shares_outstanding
        self.dividend_per_share = stock_struct.dividend_per_share
        self.earnings_per_share = stock_struct.earnings_per_share
        self.quote_values = stock_struct.quote_values
        self.initial_price = stock_struct.initial_price
        self.compounding_type = stock_struct.compounding_type
        self.dividend_growth_rate_ = stock_struct.dividend_growth_rate
        self.dividend_compounding_frequency = stock_struct.dividend_compounding_frequency
        # FinancialInstrumentStruct parameters
        self.instrument_type = financial_instrument_struct.instrument_type
        self.asset_class = financial_instrument_struct.asset_class
        self.identifier = financial_instrument_struct.identifier
        # PortfolioInstrumentStruct parameters
        self.portfolio_price_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_time_array = portfolio_instrument_struct.portfolio_price_array
        self.portfolio_predictable_income = portfolio_instrument_struct.portfolio_predicatble_income
    
    def __apply_value_quotation_type(self, values: np.ndarray) -> np.ndarray:
        """
        ## Description
        Converts values from values per share to total value.
        ### Input:
            - values (np.ndarray): Array of values to convert
        ### Output:
            - Converted values in either PER_SHARE or TOTAL format (np.ndarray)
        """
        type_check(value=values, type_=np.ndarray, value_name="values")
        match self.quote_values:
            case QuoteValues.PER_SHARE:
                return values
            case QuoteValues.TOTAL:
                return values * self.n_shares_outstanding
    
    def present_value(self, stock_valuation_method: StockValuationType=StockValuationType.DIVIDEND_DISCOUNT_MODEL,
                      # Dividend discount model
                      expected_dividend: Union[float, None]=None,
                      # Valuation by comparables
                      pe: float=0.0, pb: float=0.0, ev_to_ebitda: float=0.0, valuation_params: Union[ValuationByComparablesParams, None]=None) -> float:
        """
        ## Descdription
        Present value of the stock.
        ### Input:
            - stock_valuation_method (StockValuationType): Method for performing stock valuation (i.e., DIVIDEND_DISCOUNT_MODEL or VALUATION_BY_COMPARABLES)
            - expected_dividend (Union[float, None]): Expected dividend amount to be earned in the future (Used for computing cost of equity capital in the DIVIDEND_DISCOUNT_MODEL)
            - pe (float): Current P/E ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - pb (float): Current P/B ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - ev_to_ebitda (float): Current EV/EBITDA ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - valuation_params (Union[ValuationByComparablesParams, None]): Parameters of similar stocks (Used for VALUATION_BU_COMPARABLES)
        ### Output:
            - Present value (float) of the stock
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Dividend_discount_model, https://en.wikipedia.org/wiki/Valuation_using_multiples
            - Original Source: N/A
        """
        type_check(value=stock_valuation_method, type_=StockValuationType, value_name="stock_valuation_method")
        match stock_valuation_method:
            case StockValuationType.DIVIDEND_DISCOUNT_MODEL:
                return self.dividend_discount_model(expected_dividend=expected_dividend)
            case StockValuationType.VALUATION_BY_COMPARABLES:
                if isinstance(valuation_params, ValuationByComparablesParams) is False:
                    raise TypeError("For the VALUATION_BY_COMPARABLES the argument valuation_params must be defined as ValuationByComparablesParams type.")
                return self.valuation_by_comparables(pe=pe, pb=pb, ev_to_ebitda=ev_to_ebitda, valuation_params=valuation_params, verbose=False)
    
    def net_present_value(self, stock_valuation_method: StockValuationType=StockValuationType.DIVIDEND_DISCOUNT_MODEL,
                          # Dividend discount model
                          expected_dividend: Union[float, None]=None,
                          # Valuation by comparables
                          pe: float=0.0, pb: float=0.0, ev_to_ebitda: float=0.0, valuation_params: Union[ValuationByComparablesParams, None]=None) -> float:
        """
        ## Descdription
        Net present value of the stock.
        ### Input:
            - stock_valuation_method (StockValuationType): Method for performing stock valuation (i.e., DIVIDEND_DISCOUNT_MODEL or VALUATION_BY_COMPARABLES)
            - expected_dividend (Union[float, None]): Expected dividend amount to be earned in the future (Used for computing cost of equity capital in the DIVIDEND_DISCOUNT_MODEL)
            - pe (float): Current P/E ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - pb (float): Current P/B ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - ev_to_ebitda (float): Current EV/EBITDA ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - valuation_params (Union[ValuationByComparablesParams, None]): Parameters of similar stocks (Used for VALUATION_BU_COMPARABLES)
        ### Output:
            - Present value of the stock minus the initial price it took to purchase the stock (float)
        """
        return -self.initial_price + self.present_value(stock_valuation_method=stock_valuation_method, expected_dividend=expected_dividend,
                                                        pe=pe, pb=pb, ev_to_ebitda=ev_to_ebitda, valuation_params=valuation_params)
    
    def future_value(self, time: float, stock_valuation_method: StockValuationType=StockValuationType.DIVIDEND_DISCOUNT_MODEL,
                     # Dividend discount model
                     expected_dividend: Union[float, None]=None,
                     # Valuation by comparables
                     pe: float=0.0, pb: float=0.0, ev_to_ebitda: float=0.0, valuation_params: Union[ValuationByComparablesParams, None]=None) -> float:
        """
        ## Descdription
        Future value of the stock.
        ### Input:
            - time (float): Time at which the future value is computed
            - stock_valuation_method (StockValuationType): Method for performing stock valuation (i.e., DIVIDEND_DISCOUNT_MODEL or VALUATION_BY_COMPARABLES)
            - expected_dividend (Union[float, None]): Expected dividend amount to be earned in the future (Used for computing cost of equity capital in the DIVIDEND_DISCOUNT_MODEL)
            - pe (float): Current P/E ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - pb (float): Current P/B ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - ev_to_ebitda (float): Current EV/EBITDA ratio of the stock (Used for VALUATION_BY_COMPARABLES)
            - valuation_params (Union[ValuationByComparablesParams, None]): Parameters of similar stocks (Used for VALUATION_BU_COMPARABLES)
        ### Output:
            - Future value (float) of the stock at the given time (Computed from the present value of the stock)
        """
        # Arguments validation
        type_check(value=stock_valuation_method, type_=StockValuationType, value_name="stock_valuation_method")
        # Future value encounting multiplier
        r = self.cost_of_equity_capital(expected_dividend=expected_dividend)
        discount_term = Compounding(rate=r, compounding_type=self.compounding_type, compounding_frequency=self.dividend_compounding_frequency)
        # Present value of the stock
        pv = self.present_value(stock_valuation_method=stock_valuation_method, expected_dividend=expected_dividend, pe=pe, pb=pb,
                                ev_to_ebitda=ev_to_ebitda, valuation_params=valuation_params)
        return pv / discount_term.compounding_term(time=float(time))
    
    def cost_of_equity_capital(self, expected_dividend: Union[float, None]=None) -> float:
        """
        ## Description
        Computes the cost of equity capital (Market capitalization rate).\n
        Cost of Equity Capital = (Expected Dividend / Current Share Price) + Sustainable Growth Rate\n
        Note: It is assumed that the sustainable growth rate is the dividend growth rate.
        ### Input:
            - expected_dividend (Union[float, None]): Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
        ### Output:
            - Cost of equity capital (Market capitalization rate) (float)
        ### LaTeX Formula:
            - r = \\frac{D_{E}}{P} + g
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Cost_of_capital
            - Original Source: https://www.jstor.org/stable/1809766
        """
        if isinstance(expected_dividend, type(None)):
            expected_dividend = self.dividend_per_share
        else:
            expected_dividend = float(expected_dividend)
        return expected_dividend/self.price_per_share + self.dividend_growth_rate_
    
    def dividend_discount_model(self, expected_dividend: Union[float, None]=None) -> float:
        """
        ## Discription
        Dividend discount model evaluating the price of the stock.\n
        Note: This model assumes that the dividend cashflow grows with the rate dividend_growth_rate.
        ### Input:
            - expected_dividend (Union[float, None]): Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
        ### Output:
            - Present value (float) of the stock based on the dividend discount model
        ### LaTeX Formula:
            - \\textit{PV(Share)} = \\sum^{\\infty}_{t=1} \\frac{D_{t}}{(1 + r)^{t}}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Dividend_discount_model
            - Original Source: https://doi.org/10.2307/1927792
        """
        cost_of_equity_capital = self.cost_of_equity_capital(expected_dividend=expected_dividend)
        dividend_perpetuity = Perpetuity(perpetuity_cashflow=self.dividend_per_share, rate=cost_of_equity_capital,
                                         perpetuity_growth_rate=self.dividend_growth_rate_, compounding_type=self.compounding_type)
        pv = dividend_perpetuity.present_value()
        return self.__apply_value_quotation_type(values=np.array([pv]))[0]
    
    def valuation_by_comparables(self, pe: float, pb: float, ev_to_ebitda: float, valuation_params: ValuationByComparablesParams,
                                 verbose: bool=False) -> float:
        """
        ## Description
        Valuation of the stock by comparing its features to the features of similar stocks.
        ### Input:
            - pe (float): P/E ratio of the stock
            - pb (float): P/B ratio of the stock
            - ev_to_ebitda (float): EV/EBITDA ratio of the stock
            - valuation_params (ValuationByComparablesParams): Parameters of similar stocks that will be used for the valuation by compaison
        ### Output:
            - Valuation (float) of the stock (Price per share)
        """
        # Arguments validation
        type_check(value=valuation_params, type_=ValuationByComparablesParams, value_name="valuation_params")
        pe = float(pe)
        pb = float(pb)
        ev_to_ebitda = float(ev_to_ebitda)
        # Linear regression training
        xdata = [valuation_params.pe_ratios, valuation_params.pb_ratios, valuation_params.ev_to_ebitda]
        def line(x: np.ndarray[Any, np.ndarray], alpha: float, beta_pe: float, beta_pb: float, beta_ev_to_ebitda: float):
            return (float(alpha) + float(beta_pe)*x[0] + float(beta_pb)*x[1] + float(beta_ev_to_ebitda)*x[2])
        popt, _ = scipy.optimize.curve_fit(line, np.array(xdata), valuation_params.valuations)
        # Verbose parameters
        if verbose:
            print("alpha: {}\n beta_pe: {}\n beta_pb: {}\n beta_ev_to_ebitda: {}".format(popt[0], popt[1], popt[2], popt[3]))
        # Valuation based on trained parameters
        price_per_share = (popt[0] + popt[1]*pe + popt[2]*pb + popt[3]*ev_to_ebitda) / self.n_shares_outstanding
        return self.__apply_value_quotation_type(values=np.array([price_per_share]))[0]

    def dividend_growth_rate(self, plowback_ratio: float, roe: float, in_place: bool=False) -> float:
        """
        ## Description
        Computes dividend growth rate.\n
        Dividend Growth Rate = Plowback Ratio * ROE
        ### Input:
            - plowback_ratio (float): Plowback ratio of the stock
            - roe (float): Return on equity (ROE) of the stock
            - in_place (bool): Update the dividend growth rate of the class instance with the result
        ### Output:
            - Dividend growth rate (float)
        ### LaTeX Formula:
            - \\textit{Dividend Growth Rate} = b*ROE
        """
        g = float(plowback_ratio) * float(roe)
        if bool(in_place):
            self.dividend_growth_rate_ = g
        return g

    def present_value_of_growth_opportunities(self, expected_earnings: Union[float, None]=None,
                                              expected_dividend: Union[float, None]=None) -> float:
        """
        ## Description
        Computes present value of growth opportunities (PVGO) which corresponds to the component of stock's valuation responsible for earnings growth.\n
        PVGO = Share Price - Earnings per Share / Cost of Equity Capital
        ### Input:
            - expected_earnings (Union[float, None]): Expected earnings per share (If none provided, the earnings_per_share from the definition of the instance will be used instead)
            - expected_dividend (Union[float, None]): Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
        ### Output:
            - Present value of growth opportunities (PVGO) (float)
        ### LaTeX Formula:
            - PVGO = P - \\frac{E_{E}}{r}
        ### Links:
            - Wikipedia: https://en.wikipedia.org/wiki/Present_value_of_growth_opportunities
            - Original Source: N/A
        """
        if isinstance(expected_earnings, type(None)):
            expected_earnings = self.earnings_per_share
        else:
            expected_earnings = float(expected_earnings)
        r = self.cost_of_equity_capital(expected_dividend=expected_dividend)
        return self.price_per_share - expected_earnings/r