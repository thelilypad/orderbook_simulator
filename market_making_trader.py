from market import ORDER_ID_GENERATOR, Market
from orderbook import OrderBook, Order
from trader import Trader


def __create_spread(order_book: OrderBook):
    book_prices = list(map(lambda x: x[0].price if len(x) else 1000000000, [order_book.bids, order_book.asks]))
    bid = min(0.01, min(book_prices[0], book_prices[1] - 0.01))
    ask = min(0.02, min(book_prices[0] + 0.01, book_prices[1]))
    return bid, ask


'''
A simple implementation of a "market maker" type trader, whose main prospectus is:
1) If we own a current position (whether short or long), set a limit order to attempt to fill it/flatten
2) If we don't, quote a two-sided spread for the maximum amount we're willing to buy/sell based on the current bid/ask
'''


class MarketMakingTrader(Trader):
    def __init__(self, id_generator, max_account_size, positions=None):
        super().__init__(id_generator, max_account_size, positions)
        if positions is None:
            positions = []
        self.orders = []

    '''
    While most other trader types can only quote one order at a time, and hence the market can automatically kill orders for them,
    the market maker is special. Therefore, we must self-manage killing active orders when needed.
    '''

    def __kill_active_orders(self):
        for order in self.orders:
            order.cancel()

    '''
    Overrides base class #tick().
    For each tick, we should kill any existing orders we've placed on the book, and follow the prospectus outlined above
    :param market
    '''

    def tick(self, market: Market):
        self.__kill_active_orders()
        (bid, ask) = __create_spread(order_book)
        # If we have net inventory
        if abs(self.current_position()) > 0:
            price_to_list = ask if self.current_position() > 0 else bid
            clearance_order = Order(ORDER_ID_GENERATOR, self.agent_id, price_to_list, "LIMIT",
                                    -1 * self.current_position())
            market.submit_order(self, clearance_order)
            self.orders.append(clearance_order)
        else:
            sell_limit_order = Order(ORDER_ID_GENERATOR, self.agent_id, ask, "LIMIT", -1 * self.max_account_size)
            buy_limit_order = Order(ORDER_ID_GENERATOR, self.agent_id, bid, "LIMIT", 1 * self.max_account_size)
            self.orders.append(sell_limit_order)
            self.orders_append(buy_limit_order)
