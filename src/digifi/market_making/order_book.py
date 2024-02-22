# TODO: Add order book algorithm



def volume_imbalance(best_bid_volume: int, best_ask_volume: int) -> float:
    """
    ## Description
    The difference between the best bid and best ask quotes divided by their sum.\n
    Volume Imbalance = (Best Bid Volume - Best Ask Volume) / (Best Bid Volume + Best Ask Volume)
    ### Input:
        - best_bid_volume (int): Volume of the best bid price on the bid side of the order book
        - best_ask_volume (int): Volume of the best ask price on the ask side of the order book
    ### Output:
        - Volume imbalance (float) of the order book
    ### LaTeX Formula:
        - Imb_{t} = \\frac{V^{b}_{t}-V^{a}_{t}}{V^{b}_{t}+V^{a}_{t}}
    ### Links:
        - Wikipedia: N/A
        - Original Source: https://davidsevangelista.github.io/post/basic_statistics_order_imbalance/#:~:text=The%20Order%20Book%20Imbalance%20is,at%20the%20best%20ask%2C%20respectively.
    """
    best_bid_volume = int(best_bid_volume)
    best_ask_volume = int(best_ask_volume)
    return (best_bid_volume - best_ask_volume) / (best_bid_volume + best_ask_volume)