from typing import Union
import numpy as np
from src.digifi.utilities.general_utils import type_check



def dol(quantity_of_goods: float, price_per_unit: float, variable_cost_per_unit: float, total_fixed_cost: float) -> float:
    """
    Degree of operating leverage = (% change in profits) / (% change in sales).
    """
    dol = 1 + float(total_fixed_cost)/(float(quantity_of_goods)*(float(price_per_unit)-float(variable_cost_per_unit))-float(total_fixed_cost))
    return  dol



def pe_ratio(share_price: float, eps: float) -> float:
    """
    Price-to-earnings ratio = (share price) / (earnings per share).
    """
    return float(share_price)/float(eps)



def dividend_yield(share_price: float, dividend: float) -> float:
    """
    Dividend yield = 100 * (dividend) / (share price).
    """
    return 100*float(dividend)/float(share_price)



def book_value(assets: float, liabilities: float) -> float:
    """
    Book value = assets - liabilities.
    """
    return float(assets) - float(liabilities)



def pb_ratio(market_cap: float, book_value: float) -> float:
    """
    Price-to-book ratio = (market capitalization) / (book value).
    """
    return float(market_cap)/float(book_value)



def cost_of_equity_capital(share_price: float, expected_dividend: float, expected_share_price: float) -> float:
    """
    Cost of equity capital = (expected dividend + expected share price - share price) / share price
    """
    return (float(expected_dividend) + float(expected_share_price) - float(share_price))/float(share_price)



def maximum_drawdown(returns: Union[list, np.ndarray]) -> float:
    """
    Maximum drawdown = (trough value - peak value) / peak value.
    """
    type_check(value=returns, type_=np.ndarray, value_name="returns")
    maxmimum_drawdown_candidates = [0]
    minimum = returns[0]
    maximum = returns[0]
    for i in returns:
        if i<minimum:
            minimum = i
        elif maximum<i:
            maximum = i
        else:
            if maximum==0:
                continue
            maxmimum_drawdown_candidates.append((minimum-maximum)/maximum)
            minimum = i
            maximum = i
    return float(np.max(maxmimum_drawdown_candidates))