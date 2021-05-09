import numpy
from orderbook import Order
from trader import Trader
from market import ORDER_ID_GENERATOR, Market

'''
The basic implementation of the liquidating, random trader.
1) For each turn, the liquidating random trader should decide whether to add to a long position (with variable probability) OR
2) If the current unrealized PnL of the entire open position exceeds some loss threshold, dump all open positions with a market order
'''


class LiquidatingRandomTrader(Trader):
    def __init__(self, id_generator, max_account_size, loss_threshold, positions=None):
        if positions is None:
            positions = []
        super().__init__(id_generator, max_account_size, positions)
        self.loss_threshold = loss_threshold
        self.should_continue_to_liquidate = False

    def __liquidate_position(self, market):
        market_order = self.create_order(ORDER_ID_GENERATOR, None, "MARKET", -1 * self.current_position())
        if self.current_position() == 0:
            self.should_continue_to_liquidate = False
            return
        try:
            market.submit_order(self, market_order)
        except Exception:
            # Wait until some liquidity appears
            return
        if market_order.order_state == 'CANCEL_PARTIAL_UNFILLED':
            self.should_continue_to_liquidate = True

    def __random_buy_at_market(self, market: Market):
        order_size = numpy.random.choice([0, 10, 100, 1000], p=[0.9, 0.07, 0.02, 0.01])
        # cur_pos: -100
        # max_size: -100
        # order_size: -100
        additional_size = min(abs(order_size), self.max_account_size - self.current_position()) * numpy.sign(order_size)
        if additional_size > 0:
            market_order = self.create_order(ORDER_ID_GENERATOR, None, "MARKET", additional_size)
            try:
                market.submit_order(self, market_order)
            except Exception:
                # Wait until some liquidity appears
                return

    def tick(self, market: Market):
        best_spot = market.get_current_spot().best_spot_price()
        if best_spot and self.unrealized_pnl(best_spot) < self.loss_threshold or self.should_continue_to_liquidate:
            self.__liquidate_position(market)
        else:
            self.__random_buy_at_market(market)
