import random
from orderbook import OrderBook, next_id, Order
from typing import Optional, List

ORDER_ID_GENERATOR = next_id()
AGENT_ID_GENERATOR = next_id()

'''
Simple method for gating in various traders who can have multiple quotes active on the book
This is de facto required to support market makers, who will quote both sides of the spread
:param trader: a trader (subclass of Trader)
'''


def can_trader_have_multiple_active_quotes(trader: 'Trader'):
    return type(trader).__name__ in ['MarketMakingTrader']


class SpotPrices:
    def __init__(self, bid: Optional[float], ask: Optional[float], last_transacted_price: Optional[float]):
        self.bid = bid
        self.ask = ask
        self.last_transacted_price = last_transacted_price

    def best_spot_price(self) -> Optional[float]:
        if self.ask:
            return self.ask
        return self.last_transacted_price


'''
Implementation of the Market class, which provides an abstraction of the marketplace (versus the orderbook which ostensibly handles orders)
This is required to:
- Manage various trader types (e.g. the "market" should know who is a market maker, while the orderbook shouldn't)
- Manage the market session (abstracted in this sim as "turns"/iterations)
'''


class Market:
    def __init__(self, traders=None, max_iterations=50000):
        if traders is None:
            traders = []
        self.traders = traders
        self.max_iterations = max_iterations
        self.orderbook = OrderBook()

    def add_trader(self, trader: 'Trader'):
        self.traders.append(trader)

    def remove_trader(self, trader: 'Trader'):
        self.traders.remove(trader)

    def get_current_spot(self) -> SpotPrices:
        highest_bid = self.orderbook.bids[0].price if self.orderbook.bids else None
        lowest_ask = self.orderbook.asks[0].price if self.orderbook.asks else None
        last_price = self.orderbook.last_transacted_price
        return SpotPrices(highest_bid, lowest_ask, last_price)

    '''
    Helper method to provide an initial state to the market/book
    This is probably required to set an initial 'price' for the asset
    :param orders: a list of orders (instance of Order)
    '''

    def construct_initial_state(self, orders: List[Order]):
        for order in orders:
            self.orderbook.add_order(order)

    '''
    Accessor method for traders to submit orders to the marketplace (they can directly cancel and modify orders)
    This validates that market orders are canceled if there are no orders to match it, and also ensures various traders
    can only submit quotes according to their type
    :param trader: an trader (subclass of Trader)
    :param order: an order (instance of Order)
    '''

    def submit_order(self, trader: 'Trader', order: Order):
        # For market orders, we need to reject the order if there aren't any orders available on the book to fill it
        has_liquidity_available = self.orderbook.has_active_asks() if order.buy_or_sell() == 'BUY' else self.orderbook.has_active_bids()

        # If the trader submits a market order and there's no bids or asks (depending on type), we should throw an error
        if order.order_type == 'MARKET' and not has_liquidity_available:
            raise Exception("No available liquidity to fill market order")
        # Only certain trader agents like market makers should be able to have multiple active quotes at a time
        # They must also implement their own methods for cancelling orders
        if not can_trader_have_multiple_active_quotes(trader):
            # Each agent should only be able to have one active order at a time. If there are existing ones, cancel them
            self.orderbook.attempt_kill_orders_for_agent(order.agent_id)
        # Finally submit the order to the book
        self.orderbook.add_order(order)

    '''
    This is the "clock" of our simulator, allowing us to segment market interactions into "turns"
    On each turn, we shuffle the order of traders executing a decision (to simulate fairly random movement)
    '''

    def run(self):
        for tick in range(self.max_iterations):
            randomized_trader_order = self.traders
            random.shuffle(randomized_trader_order)
            for trader in randomized_trader_order:
                trader.tick(self.orderbook)
