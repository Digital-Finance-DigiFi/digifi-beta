import numpy as np
from src.digifi.utilities.general_utils import compare_array_len



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



def bootstrap(principals: np.ndarray, maturities: np.ndarray, coupons: np.ndarray, prices: np.ndarray, coupon_dt: np.ndarray) -> np.ndarray:
    """
    Spot rate computation for a given list of bonds.
    The argument coupon_dt is the difference between times of coupon payments (e.g., for semi-annual coupon coupon_dt=0.5).
    """
    compare_array_len(array_1=principals, array_2=maturities, array_1_name="principals", array_2_name="maturities")
    compare_array_len(array_1=principals, array_2=coupons, array_1_name="principals", array_2_name="coupons")
    compare_array_len(array_1=principals, array_2=prices, array_1_name="principals", array_2_name="prices")
    compare_array_len(array_1=principals, array_2=coupon_dt, array_1_name="principals", array_2_name="coupon_dt")
    if sum(coupon_dt>1)+sum(coupon_dt<0)!=0:
        raise ValueError("The argument coupon_dt must have entries defined within the [0, 1] range.")
    spot_rates = np.array([])
    for i in range(len(principals)):
        payment_times_before_maturity = np.arange(start=coupon_dt[i], stop=maturities[i], step=coupon_dt[i])
        discount_term = 0
        for time_step in payment_times_before_maturity:
            discount_term = discount_term + np.exp(-time_step*spot_rates[np.nonzero(maturities==time_step)][0])
        spot_rate = -np.log((prices[i] - coupons[i]*coupon_dt[i]*discount_term)/(principals[i] + coupons[i]*coupon_dt[i]))/maturities[i]
        spot_rates = np.append(spot_rates, spot_rate)
    return spot_rates