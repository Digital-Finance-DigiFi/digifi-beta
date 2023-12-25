from typing import List
from src.digifi.financial_instruments.bonds import Bond
import numpy as np



def bond_price_from_yield(current_price: float, duration: float, convexity: float, yield_change: float) -> float:
    """
    Bond pricing via Taylor series expansion of bond price assuming it only depends on the yield.
    B_{t}-B_{t-1} = \Delta B_{t} = \\frac{dB}{dy}\\Delta y + 0.5\\frac{d^{2}B}{dy^{2}}\Delta y^{2}
    """
    current_price = float(current_price)
    yield_change = float(yield_change)
    db_dy = -current_price*float(duration)
    d2b_dy2 = float(convexity)*current_price
    future_bond_price = current_price + db_dy*(yield_change) + (1/2)*d2b_dy2*(yield_change)**2
    return future_bond_price



def bootstrap(bonds: List[Bond]) -> np.ndarray:
    """
    Spot rate computation for a given list of bonds.
    """
    # TODO: Add code for the bootstrap method
    return np.ones(1)