from typing import Union
import numpy as np
from src.digifi.utilities.general_utils import type_check



def maximum_drawdown(returns: Union[list, np.ndarray]) -> float:
    """
    ## Description
    Maximum drawdown = (trough value - peak value) / peak value.
    ## Links
    - Wikipedia: https://en.wikipedia.org/wiki/Drawdown_(economics)
    - Original Source: N/A
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