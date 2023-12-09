import numpy as np



def dol(quantity_of_goods: float, price_per_unit: float, variable_cost_per_unit: float, total_fixed_cost: float) -> float:
    """
    Degree of operating leverage = (% change in profits) / (% change in sales).
    """
    dol = 1+total_fixed_cost/(quantity_of_goods*(price_per_unit-variable_cost_per_unit)-total_fixed_cost)
    return  dol



def pe_ratio(share_price: float, eps: float) -> float:
    """
    Price-to-earnings ratio = (share price) / (earnings per share).
    """
    return share_price/eps



def dividend_yield(share_price: float, dividend: float) -> float:
    """
    Dividend yield = 100 * (dividend) / (share price).
    """
    return 100*dividend/share_price



def book_value(assets: float, liabilities: float) -> float:
    """
    Book value = assets - liabilities.
    """
    return assets-liabilities



def pb_ratio(market_cap: float, book_value: float) -> float:
    """
    Price-to-book ratio = (market capitalization) / (book value).
    """
    return market_cap/book_value