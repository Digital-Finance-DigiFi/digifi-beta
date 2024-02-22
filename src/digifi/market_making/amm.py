from dataclasses import dataclass
import numpy as np
from digifi.utilities.general_utils import (type_check, DataClassValidation)



@dataclass(slots=True)
class AMMToken(DataClassValidation):
    """
    ## Description
    Token data format used to define AMMLiquidityPool.
    ### Input:
        - identifier (str): Token identifier/name
        - supply (float): Supply of token in the liquidity pool
        - fee_lower_bound (float): Lower bound for possible fee
        - fee_upper_bound (float): Upper bound for possible fee
    """
    identifier: str
    supply: float
    fee_lower_bound: float
    fee_upper_bound: float



@dataclass(slots=True)
class AMMLiquidityPool(DataClassValidation):
    """
    ## Description
    Liquidity data for an AMM.\n
    Characteristic Number = Token 1 Supply * Token 2 Supply
    ### Input:
        - token_1 (AMMToken): Token 1 data
        - token_2 (AMMToken): Token 2 data
        - char_number (float): Characteristic number
    ### LaTeX Formula:
        - \\textit{Characteristic Number} = S_{1}\\times S_{2}
    """
    token_1: AMMToken
    token_2: AMMToken
    char_number: float

    def validate_char_number(self, value: float, **_) -> float:
        if value<=0:
            raise ValueError("The argument char_number must be positive.")
        return value
    
    def __post_init__(self) -> None:
        super(AMMLiquidityPool, self).__post_init__()
        if self.token_1.supply*self.token_2.supply!=self.char_number:
            raise ValueError("The argument char_number must be the product of supplies of the tokens.")



@dataclass(slots=True)
class AMMTransactionData(DataClassValidation):
    """
    ## Description
    Transaction data used to pass transactions into AMM methods.
    ### Input:
        - token_id (str): Token identifier/name that is being purchased
        - quantity (float): Number of tokens to purchase from exchange
        - percent_fee (float): Fee size as the percentage of transaction
    """
    token_id: str
    quantity: float
    percent_fee: float

    def validate_quantity(self, value: float, **_) -> float:
        if value<=0:
            raise ValueError("The argument quantity must be positive.")
        return value



class SimpleAMM:
    """
    ## Description
    Contains computational methods for an AMM with the liquidity pool given by\n
    Characteristic Number = Token 1 Supply * Token 2 Supply
    ### Input:
        - initial_liquidity_pool (AMMLiquidityPool): State of the liquidity pool to initiate the AMM with
    ### LaTeX Formula:
        - \\textit{Characteristic Number} = S_{1}\\times S_{2}
    ### Links:
        - Wikipedia: N/A
        - Original Source: https://doi.org/10.48550/arXiv.2106.14404
    """
    def __init__(self, initial_liquidity_pool: AMMLiquidityPool) -> None:
        # SimpleAMMLiquidity parameters
        self.token_1 = initial_liquidity_pool.token_1
        self.token_2 = initial_liquidity_pool.token_2
        self.char_number = initial_liquidity_pool.char_number
    
    def make_transaction(self, tx_data: AMMTransactionData) -> dict[str, float]:
        """
        ## Description
        Buy a quntity of a token from the AMM by submitting the buy order quoted in terms of the token to putchase.\n
        Transaction includes fee as the percentage of the quantity purchased.
        ### Input:
            - tx_data (AMMTransactionData): Transaction data for AMM to process
        ### Output:
            - quantity_to_sell (float): Amount of token that has to be sold to the AMM in exchange for the token being purchased
            - exchange_price (float): Exchange rate produced by the AMM
            - fee_in_purchased_token (float): Transaction fee that has to be paid quoted in quantity of purchased token (e.g., fee is 2.1 Purchased Tokens)
        """
        # Arguments validation
        type_check(value=tx_data, type_=AMMTransactionData, value_name="tx_data")
        if tx_data.token_id==self.token_1.identifier:
            token = self.token_1
            other_token = self.token_2
        elif tx_data.token_id==self.token_2.identifier:
            token = self.token_2
            other_token = self.token_1
        else:
            raise ValueError("The token with identifier {} does not exist in SimpleAMM.".format(tx_data.token_id))
        tx_buy_size = tx_data.quantity * (1+tx_data.percent_fee)
        if token.supply<tx_buy_size:
            raise ValueError("Not enough supply of token {} ({}) to fill in the buy order of {}.".format(token.identifier, token.supply, tx_buy_size))
        if (tx_data.percent_fee<token.fee_lower_bound) or (token.fee_upper_bound<tx_data.percent_fee):
            raise ValueError("The argument percent_fee must be in the range [{}, {}].".format(token.fee_lower_bound, token.fee_upper_bound))
        # Change in supply of token (y - delta_y)
        updated_token_supply = token.supply - tx_buy_size
        # Update supply of other_token based on the characteristic number (x + delta_x = K/(y - delta_y))
        updated_other_token_supply = self.char_number / updated_token_supply
        # Determine amount of other_token that needs to be sold to AMM to fill the token buy order (delta_x)
        dx = updated_other_token_supply - other_token.supply
        # Exchange price (P = (x + delta_x)/(y - delta_y))
        price = updated_other_token_supply / updated_token_supply
        # Fee quoted in terms of token
        fee = tx_data.quantity * tx_data.percent_fee * price
        # Update liquidity pool
        token.supply = updated_token_supply
        other_token.supply = updated_other_token_supply
        return {"quantity_to_sell":dx, "exchange_price":price, "fee_in_purchased_token":fee}

    def get_liquidity_curve(self, n_points: int, token_1_start: float=0.01, token_1_end: float=100_000) -> dict[str, np.ndarray]:
        """
        ## Description
        Generates points to plot the liquidity curve of the AMM.
        ### Input:
            - n_points (int): Number of points to generate
            - token_1_start (float): Starting point of the x-axis
            - token_1_end (float): Final point of the x-axis
        ### Output:
            - x (np.ndarray): x-axis of the liquidity curve
            - y (np.ndarray): y-axis of the liquidity curve
        """
        x = np.linspace(start=(token_1_start), stop=float(token_1_end), num=int(n_points))
        y = self.char_number/x
        return {"x":x, "y":y}
        