def dol(quantity_of_goods: float, price_per_unit: float, variable_cost_per_unit: float, total_fixed_cost: float) -> float:
    """
    ## Description
    Measure of how revenue growth translates to growth of income.\n
    Degree of Operating Leverage = (% Change in Profits) / (% Change in Sales) = 1 + Total Fixed Cost / (Quantity of Goods Sold * (Price per Unit - Variable Cost per Unit) - Total Fixed Cost)
    ### Input:
        - quantity_of_goods (float): Quantity of goods sold
        - price_per_unit (float): Price of every unit of good sold
        - variable_cost_per_unit (float): Variable cost accumulated when producing a unit of good
        - total_fixed_cost (float): Total fixed costs of producing all units sold
    ### Output:
        - Degree of operating leverage (DOL) (float)
    ### LaTeX Formula:
        - DOL = 1 = \\frac{F}{Q(P-V) - F}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Operating_leverage
        - Original Source: N/A
    """
    return (1 + float(total_fixed_cost)/(float(quantity_of_goods)*(float(price_per_unit)
                                                                   -float(variable_cost_per_unit))-float(total_fixed_cost)))



def pe_ratio(share_price: float, eps: float) -> float:
    """
    ## Description
    The ratio of market price to earnings.\n
    Price-to-Earnings Ratio = Share Price / Earnings per Share
    ### Input:
        - share_price (float): Share price of the company
        - eps (float): Earnings per share of the company
    ### Output:
        - P/E ratio (float)
    ### LaTeX Formula:
        - PE = \\frac{P}{EPS}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Price%E2%80%93earnings_ratio
        - Original Source: N/A
    """
    return float(share_price)/float(eps)



def pb_ratio(market_cap: float, book_value: float) -> float:
    """
    ## Description
    The ratio of market price to book value.\n
    Price-to-Book Ratio = Market Capitalization / Book Value
    ### Input:
        - market_cap (float): Market capitalization of the company
        - book_value (float): Value of the assets minus liabilities
    ### Output:
        - PB ratio (float)
    ### LaTeX Formula:
        - PB = \\frac{Market Capitalization}{Book Value}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/P/B_ratio
        - Original Source: N/A
    """
    return float(market_cap)/float(book_value)



def dividend_yield(share_price: float, dividend: float) -> float:
    """
    ## Description
    The ratio of dividend issued by the company to share price.\n
    Dividend Yield = 100 * Dividend / Share Price
    ### Input:
        - share_price (float): Share price of the company
        - dividend (float): Amount of dividend Paid out by the company per defined period
    ### Output:
        - Dividend yield (float)
    ### LaTeX Formula:
        - D_{Y} = 100\\frac{D}{P}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Dividend_yield
        - Original Source: N/A
    """
    return 100*float(dividend)/float(share_price)



def book_value(assets: float, liabilities: float) -> float:
    """
    ## Description
    Value of assets of the company minus its liabilities.\n
    Book Value = Assets - Liabilities
    ### Input:
        - assets (float): Total assets of the company
        - liabilities (float): Total liabilities of the company
    ### Output:
        - Book value (float)
    ### LaTeX Formula:
        - \\textit{Book Value} = \\textit{Assets} - \\textit{Liabilities}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Book_value
        - Original Source: N/A
    """
    return float(assets) - float(liabilities)



def cost_of_equity_capital(share_price: float, expected_dividend: float, expected_share_price: float) -> float:
    """
    ## Description
    Cost of equity capital (Market capitalization rate).\n
    Cost of Equity Capital = (Expected Dividend + Expected Share Price - Share Price) / Share Price
    ### Input:
        - share_price (float): Share price of the company
        - expected_dividend (float): Expected dividend to be received in the future
        - expected_share_price (float): Expected share price of the company in the future
    ### Output:
        - Cost of equity capital (Market capitalization rate) (float)
    ### LaTeX Formula:
        - r = \\frac{D_{t+1} + P_{t+1} - P_{t}}{P_{t}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Cost_of_equity
        - Original Source: N/A
    """
    return (float(expected_dividend) + float(expected_share_price) - float(share_price))/float(share_price)



def roe(total_earnings: float, book_value: float) -> float:
    """
    ## Description
    Measure of profitability of the company in relation to its equity.\n
    ROE = Total Earnings / Book Value
    ### Input:
        - total_earnings (float): Total earnings of the company
        - book_value (float): Value of the assets minus liabilities
    ### Output:
        - Return on equity (ROE)
    ### LaTeX Formula:
        - ROE  =\\frac{\\textit{Total Earnings}}{\\textit{Book Value}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Return_on_equity
        - Original Source: N/A
    """
    return float(total_earnings) / float(book_value)



def payout_ratio(dividend_per_share: float, earnings_per_share: float) -> float:
    """
    ## Description
    Ratio of dividends to earnings per share.\n
    Payout Ratio = Dividend per Share / Earnings per Share
    ### Input:
        - dividend_per_share (float): Dividend per share paid out closest to the latest earnings
        - earnings_per_share (float): Earnings per share
    ### Output:
        - Payout ratio (float)
    ### LaTeX Formula:
        - \\textit{Payout Ratio} = \\frac{D_{t}}{EPS_{t}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Dividend_payout_ratio
        - Original Source: N/A
    """
    return float(dividend_per_share) / float(earnings_per_share)



def plowback_ratio(dividend_per_share: float, earnings_per_share: float) -> float:
    """
    ## Description
    One minus payout ratio.\n
    Plowback Ratio = 1 - (Dividend per Share / Earnings per Share)
    ### Input:
        - dividend_per_share (float): Dividend per share paid out closest to the latest earnings
        - earnings_per_share (float): Earnings per share
    ### Output:
        - Plowback ratio (float)
    ### LaTeX Formula:
        - \\textit{Plowback Ratio} = 1 - \\frac{D_{t}}{EPS_{t}}
    """
    return 1 - float(dividend_per_share) / float(earnings_per_share)



def altman_z_score(EBIT: float, total_assets:float, sales: float, equity: float, total_liabilities: float,
                   retained_earnings: float, working_capital: float) -> float:
    """
    ## Description
    Measure for predicting the likelihood of bankrupcy of a company.
    ### Input:
        - EBIT (float): EBIT of the company
        - total_assets (float): Total assets of the company
        - sales (float): Total sales of the company
        - equity (float): Market value of equity
        - total_liabilities (float): Total liabilities of the company
        - retained_earnings (float): Retained earnings of the company
        - working_capital (float): Working capital of the company
    ### Output:
        - Altman's Z-Score (float): If the value is below 1.81 - there is a high vulnerability to bankrupcy,
        if the value is above 2.99 - there is a low vulnerability to bankrupcy
    ### LaTeX Formula:
        - Z = 3.3\\frac{EBIT}{\\textit{Total Assets}} + 1.0\\frac{Sales}{Assets} + 0.6\\frac{Equity}{\\textit{Total Liabilities}} + 
    1.4\\frac{\\textit{Retained Earning}}{\\textit{Total Assets}} + 1.2\\frac{\\textit{Working Capital}}{\\textit{Total Assets}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Altman_Z-score
        - Origina Source: https://doi.org/10.1002/9781118266236.ch19
    """
    total_assets = float(total_assets)
    return (3.3*float(EBIT)/float(total_assets) + float(sales)/total_assets + 0.6*float(equity)/float(total_liabilities)
            + 1.4*float(retained_earnings)/total_assets + 1.2*float(working_capital)/total_assets)



def weighted_average_cost_of_capital(equity: float, debt: float, return_on_equity: float, return_on_debt: float, corporate_tax: float=0.0) -> float:
    """
    ## Description
    Computes the weighted average cost of capital (WACC), which is the expected return on the company's assets.\n
    WACC = (Debt / (Debt+Equity) * (1 - Corporate Tax Rate) * Return on Debt) + (Equity / (Debt+Equity) * Return on Equity)
    ### Input:
        - equity (float): Total equity of the company
        - debt (float): Total debt of the company
        - return_on_equity (float): Expected return on equity of the company
        - return_on_debt (float): Expected return on debt of the company
        - corporate_tax (float): Corporate tax rate on earnings after interest, EBT
    ### Output:
        - Weighted average cost of capital (WACC) (float)
    ### LaTeX Formula:
        - r_{A} = [r_{D}(1-T_{c})\\frac{D}{E+D}] + [r_{E}\\frac{E}{E+D}]
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Weighted_average_cost_of_capital
        - Original Source: N/A
    """
    equity = float(equity)
    debt = float(debt)
    return (debt/(debt+equity) * float(return_on_debt)) + (equity/(debt+equity) * float(return_on_equity))



def expected_return_on_equity(equity: float, debt: float, return_on_assets: float, return_on_debt: float, corporate_tax: float=0.0) -> float:
    """
    ## Description
    Computes expected return on the equity (ROE) portion of the company.\n
    ROE = Return on Assets + (Return on Assets - Return on Debt * (1 - Corporate Tax Rate)) * Debt / Equity
    ### Input:
        - equity (float): Total equity of the company
        - debt (float): Total debt of the company
        - return_on_assets (float): Return on all assets (WACC) of the company
        - return_on_debt (float): Expected return on debt of the company
        - corporate_tax (float): Corporate tax rate on earnings after interest, EBT
    ### Output:
        - Expected return on equity (ROE) (float)
    ### LaTeX Formula:
        - r_{E} = r_{A} + (r_{A}-r_{D}(1-T_{c}))\\frac{D}{E}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Return_on_equity
        - Original Source: N/A
    """
    return_on_assets = float(return_on_assets)
    return return_on_assets + (return_on_assets - float(return_on_debt)) * (float(debt) / float(equity))



def unlevered_beta(equity: float, debt: float, beta_equity: float, beta_debt: float) -> float:
    """
    ## Description
    Unlevered beta, which is the systematic risk of the company's assets.\n
    Unlevered Beta = (Debt / (Debt+Equity) * Beta of Debt) + (Equity / (Debt+Equity) * Beta of Equity)
    ### Input:
        - equity (float): Total equity of the company
        - debt (float): Total debt of the company
        - beta_equity (float): Levered beta of the company
        - beta_debt (float): Beta debt of the company
    ### Output:
        - Unlevered beta (float)
    ### LaTeX Formula:
        - \\beta_{A} = [\\beta_{D}\\frac{D}{E+D}] + [\\beta_{E}\\frac{E}{E+D}]
    """
    equity = float(equity)
    debt = float(debt)
    return (debt/(debt+equity) * float(beta_debt)) + (equity/(debt+equity) * float(beta_equity))



def levered_beta(equity: float, debt: float, beta_assets: float, beta_debt: float) -> float:
    """
    ## Description
    Levered beta, which is the equity-only beta of the company.\n
    Levered Beta = Beta of Assets + (Beta of Assets - Beta of Debt) * (Debt / Equity)
    ### Input:
        - equity (float): Total equity of the company
        - debt (float): Total debt of the company
        - beta_assets (float): Unlevered beta of the company
        - beta_debt (float): Beta debt of the company
    ### Output:
        - Levered beta (float)
    ### LaTeX Formula:
        - \\beta_{E} = \\beta_{A} + (\\beta_{A} - \\beta_{D})\\frac{D}{E}
    """
    beta_assets = float(beta_assets)
    return beta_assets + (beta_assets - float(beta_debt)) * float(debt) / float(equity)



def relative_tax_advantage_of_debt(corporate_tax: float, personal_tax: float, effective_personal_tax: float) -> float:
    """
    ## Description
    Calculates an advantage of debt financing for a company as opposed to equity financing from perspective of tax optimization.\n
    Relative Tax Advantage of Debt = (1 - Personal Tax on Interest Income) / ((1 - Effective Personal Tax) * (1 - Corporate Tax))
    ### Input:
        - corporate_tax (float): Corporate tax rate applied to a company after debt payout
        - personal_tax (float): Personal tax rate on a interest income
        - effective_personal_tax (float): Effective tax rate on equity income comprising personal tax on dividend income and personal tax on capital gains income
    ### Output:
        - Relative tax advantage of debt ratio (float)
    ### LaTeX Formula:
        - \\textit{Relative Tax Advantage of Debt} = \\frac{1-T_{p}}{(1-T_{pE})(1-T_{c})}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Tax_benefits_of_debt
        - Original Source: N/A
    """
    return (1 - float(personal_tax)) / ((1 - float(effective_personal_tax)) * (1 - float(corporate_tax)))