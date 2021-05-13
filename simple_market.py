import numpy as np

from liquidating_random_trader import LiquidatingRandomTrader
from market import Market, SpotPrices
from orderbook import next_id, Order
from trader import Trader
import random

TRADER_ID = next_id()

'''
Simple implementation of the market for testing basic behaviors that don't require N-body simulations
(e.g. spot price and market depth are abstracted).
In this case, instead of tracking order flow, we do the following:
- We assume market depth is static per price level (e.g. 1000 shares, for example)
- When an order occurs, it will move the price level according to how many units it represents:
   - A large buy order (e.g. 5000 shares) may exhaust 4 price levels, and push the spot up to spot + 4
   - A large sell order similarly could do the same, and decrease spot up to spot - 4 (5k shares)
- Each iteration, we can randomly change the spot price and necessarily reset the volume at that new price level.
'''


class SimpleMarket(Market):
    def __init__(self, starting_spot_price: float, base_volume_per_price_level=1000, traders=None,
                 max_iterations=50000):
        if traders is None:
            traders = []
        super().__init__(traders, max_iterations)
        self.spot_price = starting_spot_price
        self.current_volume_left_at_price = base_volume_per_price_level
        self.base_volume_per_price_level = base_volume_per_price_level
        self.traders = traders

    '''
    Simple method (can override) for changing the spot price. This should uniformly choose to modify spot by 1, -1, or 0.
    '''

    def randomly_modify_spot_price(self):
        new_spot = self.spot_price + random.randint(-1, 1)
        if new_spot != self.spot_price:
            self.current_volume_left_at_price = self.base_volume_per_price_level
        self.spot_price = new_spot

    '''
    Simple Method to bypass division by zero issues. Checking the denominator before division
    '''
    def divide_hack(n,d):
        return n / d if d else 0

    def submit_order(self, trader: Trader, order: Order):
        # This is a much more simplistic market that effectively has a set market depth per price point
        # In this case, we should effectively fill every order instantly (this doesn't really handle LIMIT
        # orders)
        # We should 'add' volume when a sell occurs at a given price level and 'subtract' volume if the order is a buy
        new_volume_at_strike = self.current_volume_left_at_price - order.order_size
        # We should bump the spread based on the differential of volume added
        # E.g. if a trader liquidates 2000 shares and our current level has 200 shares remaining
        # We should drop the price $2 and set the new volume at price level for that price to 200 shares
        spread_jumps = int(divide_hack(new_volume_at_strike, self.base_volume_per_price_level))
        # We should at the current level fill the min(order_size, current_volume_left) if it's a BUY
        # or the max of (current_volume_left - base_volume, order_size) if it's a SELL
        fill_at_current_level = min(order.order_size, self.current_volume_left_at_price) if order.order_size > 0 \
            else max(self.current_volume_left_at_price - self.base_volume_per_price_level, order.order_size)
        # We should partially execute it at the given spot without moving the price
        order.partial_execute(fill_at_current_level, self.spot_price)
        # For each completely filled order level
        for i in range(1, abs(spread_jumps)):
            # We should modify the spot price accordingly for that filled level and execute it (-1 if a sell,
            # +1 if a buy)
            self.spot_price += i * np.sign(order.order_size)
            order.partial_execute(self.base_volume_per_price_level, self.spot_price)
        # Finally, we should have left over some order to fill that hasn't completely cleaned out a price level
        order_left_to_fill = new_volume_at_strike % self.base_volume_per_price_level
        if order_left_to_fill != 0:
            # We should modify the spot level one more time
            self.spot_price += np.sign(order.order_size)
        # And execute the order
        order.partial_execute(order_left_to_fill, self.spot_price)
        cap = 0 if order.order_size < 0 else 1000
        self.base_volume_per_price_level = cap - order_left_to_fill

    '''
    Simple override to just return the spot price here.
    '''

    def get_current_spot(self) -> SpotPrices:
        return SpotPrices(self.spot_price, self.spot_price, self.spot_price)

    '''
    Overrides the base run_iteration functionality to randomly shift the spot price by +- $1 per iteration.
    If the spot price changes, we should reset the volume available for the iteration.
    '''

    def run_iteration(self, iteration: int):
        self.randomly_modify_spot_price()
        super().run_iteration(iteration)


if __name__ == "__main__":
    traders = [LiquidatingRandomTrader(TRADER_ID, 300, -10000) for i in range(200)]
    market = SimpleMarket(300, 1000, traders, 500)
    market.run()
    print(list(map(lambda x: str(x.open) + " " + str(x.close) + '\n', market.ohlcs)))
