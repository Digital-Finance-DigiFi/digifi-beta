from typing import Union
import numpy as np
from digifi.utilities.general_utils import type_check



def maximum_drawdown(returns: Union[list, np.ndarray]) -> float:
    """
    ## Description
    Measure of the decline of the asset from its historical peak.\n
    Maximum Drawdown = (Peak Value - Trough Value) / Peak Value
    ### Input:
        - returns (np.ndarray): Returns of the asset
    ### Output:
        - Maximum drawdown of the returns
    ### LaTeX Formula:
        - \\textit{Maximum Drawdown} = \\frac{\\textit{Peak Value} - \\textit{Trough Value}}{\\textit{Peak Value}}
    ### Links:
        - Wikipedia: https://en.wikipedia.org/wiki/Drawdown_(economics)
        - Original Source: N/A
    """
    # Arguments validation
    type_check(value=returns, type_=np.ndarray, value_name="returns")
    maxmimum_drawdown_candidates = [0]
    minimum = returns[0]
    maximum = returns[0]
    # Selection of maximum drawdown candidates
    for i in returns:
        if i<minimum:
            minimum = i
        elif maximum<i:
            maximum = i
        else:
            if maximum==0:
                continue
            maxmimum_drawdown_candidates.append((maximum - minimum)/maximum)
            minimum = i
            maximum = i
    return float(np.max(maxmimum_drawdown_candidates))