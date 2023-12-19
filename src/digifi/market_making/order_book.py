# TODO: Add order book algorithm



def volume_imbalance(best_bid_volume: int, best_ask_volume: int) -> float:
    """
    Imb_{t} = \\frac{V^{b}_{t}-V^{a}_{t}}{V^{b}_{t}+V^{a}_{t}}
    The difference between the best bid and best ask quotes divided by their sum.
    Original Source: https://davidsevangelista.github.io/post/basic_statistics_order_imbalance/#:~:text=The%20Order%20Book%20Imbalance%20is,at%20the%20best%20ask%2C%20respectively.
    """
    return (best_bid_volume-best_ask_volume)/(best_bid_volume+best_ask_volume)