import heapq
from collections import defaultdict
import numpy as np

"""
    Simple generator function to create order IDs sequentially.
"""


def next_id():
    i = 1
    while True:
        yield i
        i += 1


COMPLETED_ORDER_TYPES = ['CANCELLED', 'FILLED', 'CANCEL_PARTIAL_UNFILLED']
ACTIVE_ORDER_TYPES = ['ACTIVE', 'PARTIAL_EXECUTION']


class Order:
    """
    This implements the basic concept of an order for the limit book. It can be used for both buys and sells, and supports basic order types (LIMIT and MARKET)
    :param id_generator: a generator function to create a unique order ID
    :param price: The price to buy or sell order_size at.
    :param order_type: An enumeration currently supporting LIMIT or MARKET only
    :param order_size: quantity to buy (if > 0) or sell (if < 0)
    :raises Exception: This will occur if you attempt to provide a price to a market order
    """

    def __init__(self, id_generator, agent_id, price, order_type, order_size, order_fill_callback=lambda x: x):
        self.id = next(id_generator)
        self.price = price
        self.agent_id = agent_id
        self.order_size = order_size
        self.order_type = order_type
        self.order_state = 'ACTIVE'
        self.partial_execution_log = []
        self.order_fill_callback = order_fill_callback
        if self.order_type == 'MARKET' and self.price:
            raise Exception("Unable to provide price with market order")
        if self.order_type == 'LIMIT' and not self.price:
            raise Exception("Limit order requires a price to be input")

    def buy_or_sell(self):
        return "BUY" if self.order_size > 0 else "SELL"

    def cancel(self):
        self.order_state = 'CANCELLED'

    def cancel_partial_unfilled(self):
        self.order_state = 'CANCEL_PARTIAL_UNFILLED'

    def fill(self):
        self.order_state = 'FILLED'

    def is_open(self):
        return self.order_state not in COMPLETED_ORDER_TYPES

    def partial_execute(self, quantity, price):
        self.partial_execution_log.append(PartialExecution(quantity, price))
        # Partially execute the order until we hit a minimum of 0 units requested left
        self.order_size = max(abs(self.order_size - quantity), 0) * np.sign(self.order_size)
        if self.order_size != 0:
            self.order_state = 'PARTIAL_EXECUTION'
        else:
            self.order_state = 'FILLED'
        self.order_fill_callback(self)

    # Needed for proper storage in the heapq structure (there aren't keys provided, implying you need to overload <
    # to allow ordering)
    def __lt__(self, other):
        # Since heapq implements a minheap, we need to incorrectly overload SELL LIMIT orders to put the top item
        # first (the lowest ask)
        if self.order_size > 0:
            return self.price > other.price
        return self.price < other.price


class PartialExecution:
    def __init__(self, quantity, at_price):
        self.quantity = quantity
        self.at_price = at_price

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class OrderBook:
    """
    Class implementing the basic order book, including partial execution of orders, limit, and market orders.
    """

    def __init__(self):
        self.asks = []
        self.bids = []
        self.agent_orders = defaultdict(list)
        self.all_orders = {}
        self.log = []
        self.last_transacted_price = None
        heapq.heapify(self.asks)
        heapq.heapify(self.bids)

    """
    Adds an order to the matching order book (whether bid or ask). It implements the following logic:
    1) It will attempt to immediately fill the market or limit order. If the market order cannot be fully fulfilled, it will be partially executed and cancelled.
    2) Limit orders will stay on the bid/ask order books until they can be fulfilled or cancelled.
    
    :param order: an order (instance of Order)
    """

    def add_order(self, order):
        # Add it to the dict of agent orders for rapid indexing
        self.agent_orders[order.agent_id].append(order)

        # Add it to the dict of order IDs for rapid indexing
        self.all_orders[order.id] = order

        # Attempt to fill the order from the top of the book before inserting it
        has_immediately_filled = self.__attempt_fill__(order)

        # Otherwise, add it to the book based on the price
        if not has_immediately_filled:
            if order.buy_or_sell() == 'BUY':
                # Due to the minheap implementation, we have to multiply price by -1 to store it in order (highest
                # bid first)
                heapq.heappush(self.bids, order)
            else:
                heapq.heappush(self.asks, order)

    """
    Cancels an active order by the given order ID (whether or bid or ask, does not matter).
    """

    def cancel_order(self, order_id):
        order = self.all_orders[order_id]
        if not order:
            raise Exception("No order found for the given ID")
        order.cancel()

    """
    Returns the current bid-ask spread (the spread between the highest bid and lowest ask) or 0 if it does not exist (no orders on one book, for example).
    """

    def current_spread(self):
        # Remove any cancelled/filled before calculating the bid-ask spread
        self.__lazy_remove_completed_orders()

        # If there are no bids or no asks, return 0 for the spread size
        if not len(self.bids) or not len(self.asks):
            return 0.0

        # Otherwise get the lowest ask and highest bid
        highest_bid = self.bids[0]
        lowest_ask = self.asks[0]
        return lowest_ask.price - highest_bid.price

    """
    Helper for returning all active orders in asks (asks that haven't filled or been cancelled).
    """

    def has_active_asks(self):
        return len([b for b in self.asks if b.is_open()]) > 0

    """
    Helper for returning all active orders in bids (bids that haven't filled or been cancelled).
    """

    def has_active_bids(self):
        return len([b for b in self.bids if b.is_open()]) > 0

    """
    Private class method for attempting to cancel any active orders (ACTIVE or PARTIAL_EXECUTION orders).
    """

    def attempt_kill_orders_for_agent(self, agent_id):
        for order in self.agent_orders[agent_id]:
            if order.order_state in ACTIVE_ORDER_TYPES:
                order.cancel()

    """
    Private class method for lazy deletion of cancelled or fulfilled orders from the heaps. This allows us to not have to re-heapify when an order
    is cancelled and maintain speed.
    """

    def __lazy_remove_completed_orders(self):
        # Attempt to remove all cancelled or filled orders from the bid orderbook
        while len(self.bids) and not self.bids[0].is_open():
            heapq.heappop(self.bids)
        # Attempt to remove all cancelled or filled orders from the ask orderbook
        while len(self.asks) and not self.asks[0].is_open():
            heapq.heappop(self.asks)

    """
    Private class method for handling processing of orders down the tree (when an incoming order occurs).
    1) It will return early if the top of the order book is a cancelled or already filled order (and remove it)
    2) It will partially execute both the top of the order book and the current order at the quantity/price available
    3) If that exhausts the top of the order book order size, it will mark that order as fully filled
    :param order: an order (instance of Order)
    :param heap: either the bid or ask order book (depending on the sign of order_size).
    """

    def __attempt_order_match(self, order, heap):
        txn = heap[0]
        # We need to cache this so it doesn't get lost in mutations
        txn_size = abs(txn.order_size)
        order_size = abs(order.order_size)
        # If the order is already completed, toss it and continue
        if not txn.is_open():
            heapq.heappop(heap)
            return

        # The amount we fill is the minimum of the order in the book and the amount we
        # are trying to match in the current order
        fill_order_size = min(txn_size, order_size)
        txn.partial_execute(fill_order_size * np.sign(txn.order_size), txn.price)
        # If the available order is smaller or equal to our current order, pop it from the book and mark it filled
        if txn_size <= order_size:
            txn = heapq.heappop(heap)
            txn.fill()

        # Mark the price and quantity we filled the order partially at
        order.partial_execute(fill_order_size * np.sign(order.order_size), txn.price)

        # Set the last transacted price to equal the current transaction price
        # This is useful if the orderbook dries up and we need a reference
        # For the next price (e.g. 0 asks on the book, some bids)
        self.last_transacted_price = txn.price

    """
    Private class method for handling fills when an order comes in (both market and limit).
    1) It will attempt to lazy remove all orders down the heap that are already cancelled/filled
    2) It will not attempt to fill an order if the matching book is empty
    3) If the order is a market order, it will fill it (even partially) until the order size has been exhausted
    4) If the order is a market order and partially filled, it will mark it as partially filled and cancel it
    5) It will fill a limit order according to the price OR BETTER either partially or fully
    6) If the order is fully filled, it will mark it as filled
    :param order: an order (instance of Order)
    """

    def __attempt_fill__(self, order):
        # First attempt to remove any dead order types (FILLED, CANCELLED)
        self.__lazy_remove_completed_orders()

        # If order is a buy, look at asks; otherwise, look at bids
        bid_or_ask = self.asks if order.buy_or_sell() == 'BUY' else self.bids

        # If the book is empty, we can't fill it
        if not len(bid_or_ask):
            return False

        # If the order type is MARKET, attempt to aggressively fill it at any price
        if order.order_type == 'MARKET':
            while abs(order.order_size) > 0 and len(bid_or_ask):
                self.__attempt_order_match(order, bid_or_ask)
            if abs(order.order_size) > 0:
                # If the order still has remaining volume to fill and there are no more bid/asks for a market order,
                # mark it a completed state On the converse, limit orders can stay as-is
                order.cancel_partial_unfilled()
            # Always return true for a market order - either it will fill or it won't
            return True

        # If the order type is LIMIT
        else:
            # If it is a buy, try to fill it at the price provided or a lower price This is implied by the ordering
            # of the book - we check the top of the book (the lowest ask) and either it is lower/equal To our order
            # price (in which case we can fill it according to the same MARKET order logic) or it is higher (in which
            # case we have no fill)
            if order.buy_or_sell() == 'BUY':
                while len(bid_or_ask) and bid_or_ask[0].price <= order.price and abs(
                        order.order_size) > 0 and self.has_active_asks():
                    self.__attempt_order_match(order, bid_or_ask)
                    # if it is a sell, try to fill at the price provided or a higher price
            else:
                while len(bid_or_ask) and bid_or_ask[0].price >= order.price and abs(
                        order.order_size) > 0 and self.has_active_bids():
                    self.__attempt_order_match(order, bid_or_ask)

                    # If we were able to fill the entire quantity, mark the order as filled
        if order.order_size == 0:
            order.fill()
            return True
