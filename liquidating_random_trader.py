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
        super().__init__(id_generator, max_account_size, positions)
        if positions is None:
            positions = []
        self.loss_threshold = loss_threshold
        self.should_continue_to_liquidate = False

    def __liquidate_position(self, market):
        market_order = Order(ORDER_ID_GENERATOR, self.agent_id, None, "MARKET", -1 * self.current_position())
        if self.current_position() == 0:
            self.should_continue_to_liquidate = False
            return
        current_spot = market.get_current_spot()
        try:
            market.submit_order(self, market_order)
        except Error:
            # Wait until some liquidity appears
            return
        if market_order.order_state == 'CANCEL_PARTIAL_UNFILLED':
            self.should_continue_to_liquidate = True

    def __random_buy_at_market(self, market: Market):
        order_size = numpy.random.choice([0, 10, 100, 1000], p=[0.9, 0.07, 0.02, 0.01])
        additional_size = min(order_size, (self.max_position_size - (self.current_position() + order_size)))
        if additional_size > 0:
            market_order = Order(ORDER_ID_GENERATOR, self.agent_id, None, "MARKET", additional_size)
        try:
            market.submit_order(self, market_order)
        except Exception:
            # Wait until some liquidity appears
            return

    def tick(self, market: Market):
        current_asset_price = market.get_current_spot()
        if self.unrealized_pnl(current_asset_price) < self.loss_threshold or self.should_continue_to_liquidate:
            self.__liquidate_position(market)
        else:
            self.__random_buy_at_market(market)
