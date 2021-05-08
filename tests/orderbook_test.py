import os, sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orderbook import Order, OrderBook, PartialExecution, next_id

ID_GENERATOR = next_id()
AGENT_ID = 1
PRICE = 100


class TestOrder(unittest.TestCase):
    def test_buy_or_sell(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        self.assertEqual(order.buy_or_sell(), 'SELL')
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", 100)
        self.assertEqual(order.buy_or_sell(), 'BUY')

    def test_change_to_cancel(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        self.assertEqual(order.order_state, "ACTIVE")
        order.cancel()
        self.assertEqual(order.order_state, "CANCELLED")

    def test_change_to_fill(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        self.assertEqual(order.order_state, "ACTIVE")
        order.fill()
        self.assertEqual(order.order_state, "FILLED")

    def test_cannot_add_price_to_market_order(self):
        with self.assertRaises(Exception) as context:
            Order(ID_GENERATOR, AGENT_ID, PRICE, "MARKET", -100)
            self.assetEqual(context, "Unable to provide price with market order")

    def test_must_add_price_to_limit_order(self):
        with self.assertRaises(Exception) as context:
            Order(ID_GENERATOR, AGENT_ID, None, "LIMIT", -100)
            self.assetEqual(context, "Limit order requires a price to be input")

    def test_partial_execution(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        self.assertEqual(order.order_state, "ACTIVE")
        self.assertEqual(order.order_size, -100)
        order.partial_execute(20, 4)
        self.assertEqual(order.order_size, -80)
        self.assertEqual(order.order_state, "PARTIAL_EXECUTION")
        self.assertEqual(order.partial_execution_log, [PartialExecution(20, 4)])
        order.partial_execute(20, 4)
        self.assertEqual(order.order_size, -60)
        self.assertEqual(order.order_state, "PARTIAL_EXECUTION")
        self.assertEqual(order.partial_execution_log, [PartialExecution(20, 4), PartialExecution(20, 4)])
        order.partial_execute(60, 4)
        self.assertEqual(order.order_size, 0)
        self.assertEqual(order.order_state, "FILLED")
        self.assertEqual(order.partial_execution_log,
                         [PartialExecution(20, 4), PartialExecution(20, 4), PartialExecution(60, 4)])


class TestOrderBook(unittest.TestCase):
    def test_add_basic_sell_order(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        book = OrderBook()
        book.add_order(order)
        self.assertEqual(book.asks, [order])
        order1 = Order(ID_GENERATOR, 2, PRICE + 1, "LIMIT", -100)
        book.add_order(order1)
        self.assertEqual(book.asks, [order, order1])
        all_orders = {order1.id: order1, order.id: order}
        agent_orders = {order1.agent_id: [order1], order.agent_id: [order]}
        self.assertEqual(book.all_orders, all_orders)
        self.assertEqual(book.agent_orders, agent_orders)

    def test_add_basic_buy_order(self):
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", 100)
        book = OrderBook()
        book.add_order(order)
        self.assertEqual(book.bids, [order])
        order1 = Order(ID_GENERATOR, 2, PRICE + 1, "LIMIT", 100)
        book.add_order(order1)
        self.assertEqual(book.bids, [order1, order])
        all_orders = {order1.id: order1, order.id: order}
        agent_orders = {order1.agent_id: [order1], order.agent_id: [order]}
        self.assertEqual(book.agent_orders, agent_orders)
        self.assertEqual(book.all_orders, all_orders)

    def test_prevent_market_order_no_orders(self):
        order = Order(ID_GENERATOR, AGENT_ID, None, "MARKET", 100)
        book = OrderBook()
        with self.assertRaises(Exception) as context:
            book.add_order(order)
            self.assetEqual(context, "No available liquidity to fill market order")
        sell_order = Order(ID_GENERATOR, AGENT_ID, None, "MARKET", -100)
        with self.assertRaises(Exception) as context:
            book.add_order(order)
            self.assetEqual(context, "No available liquidity to fill market order")

    def test_has_active_asks(self):
        book = OrderBook()
        self.assertEqual(book.has_active_asks(), False)
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", -100)
        book.add_order(order)
        self.assertEqual(book.has_active_asks(), True)
        self.assertEqual(book.has_active_bids(), False)

    def test_has_active_bids(self):
        book = OrderBook()
        self.assertEqual(book.has_active_bids(), False)
        order = Order(ID_GENERATOR, AGENT_ID, PRICE, "LIMIT", 100)
        book.add_order(order)
        self.assertEqual(book.has_active_bids(), True)
        self.assertEqual(book.has_active_asks(), False)

    def test_has_both_active(self):
        book = OrderBook()
        self.assertEqual(book.has_active_bids(), False)
        limit_buy_order = Order(ID_GENERATOR, 1, 100, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 105, "LIMIT", -100)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        self.assertEqual(book.has_active_asks(), True)
        self.assertEqual(book.has_active_bids(), True)

    def test_should_give_correct_bid_ask(self):
        book = OrderBook()
        self.assertEqual(book.current_spread(), 0)
        limit_buy_order = Order(ID_GENERATOR, 1, 100, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 105, "LIMIT", -100)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        self.assertEqual(book.current_spread(), 5)
        limit_buy_order2 = Order(ID_GENERATOR, 3, 102, "LIMIT", 100)
        book.add_order(limit_buy_order2)
        self.assertEqual(book.current_spread(), 3)

    def test_cancel_order(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 100, "LIMIT", 100)
        book.add_order(limit_buy_order)
        self.assertEqual(limit_buy_order.order_state, 'ACTIVE')
        book.cancel_order(limit_buy_order.id)
        self.assertEqual(limit_buy_order.order_state, 'CANCELLED')

    def test_lazy_cleanup(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 100, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 105, "LIMIT", -100)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        book.cancel_order(limit_buy_order.id)
        book.cancel_order(limit_sell_order.id)
        self.assertEqual(limit_buy_order.order_state, 'CANCELLED')
        self.assertEqual(limit_sell_order.order_state, 'CANCELLED')
        self.assertEqual(len(book.bids), 1)
        self.assertEqual(len(book.asks), 1)
        book._OrderBook__lazy_remove_completed_orders()
        self.assertEqual(len(book.bids), 0)
        self.assertEqual(len(book.asks), 0)

    def test_fill_with_market_order(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 100, "LIMIT", 100)
        market_sell_order = Order(ID_GENERATOR, 2, None, "MARKET", -100)
        book.add_order(limit_buy_order)
        self.assertEqual(book.bids, [limit_buy_order])
        book.add_order(market_sell_order)
        self.assertEqual(book.bids, [])
        self.assertEqual(book.asks, [])
        self.assertEqual(market_sell_order.partial_execution_log, [PartialExecution(100, 100)])
        self.assertEqual(limit_buy_order.partial_execution_log, [PartialExecution(100, 100)])
        self.assertEqual(limit_buy_order.order_state, 'FILLED')

    def test_fill_with_limit_orders(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 200, "LIMIT", -100)
        book.add_order(limit_buy_order)
        self.assertEqual(book.bids, [limit_buy_order])
        book.add_order(limit_sell_order)
        self.assertEqual(book.bids, [])
        self.assertEqual(book.asks, [])
        self.assertEqual(limit_sell_order.partial_execution_log, [PartialExecution(100, 200)])
        self.assertEqual(limit_buy_order.partial_execution_log, [PartialExecution(100, 200)])
        self.assertEqual(limit_buy_order.order_state, 'FILLED')

    def test_fill_with_partial_limit_order(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 200, "LIMIT", -50)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        self.assertEqual(book.bids, [limit_buy_order])
        self.assertEqual(book.asks, [])
        self.assertEqual(book.bids[0].order_size, 50)
        self.assertEqual(limit_sell_order.order_size, 0)

    def test_fill_with_partial_limit_order_multiple(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 200, "LIMIT", -50)
        limit_sell_order2 = Order(ID_GENERATOR, 2, 200, "LIMIT", -25)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        book.add_order(limit_sell_order2)
        self.assertEqual(book.bids, [limit_buy_order])
        self.assertEqual(book.asks, [])
        self.assertEqual(book.bids[0].order_size, 25)
        self.assertEqual(limit_sell_order.order_size, 0)
        self.assertEqual(limit_sell_order2.order_size, 0)

    def test_fill_with_partial_limit_order_multiple_and_diff_prices(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 195, "LIMIT", -50)
        limit_sell_order2 = Order(ID_GENERATOR, 3, 200, "LIMIT", -25)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        book.add_order(limit_sell_order2)
        self.assertEqual(book.bids, [limit_buy_order])
        self.assertEqual(book.asks, [])
        self.assertEqual(book.bids[0].order_size, 25)
        self.assertEqual(limit_sell_order.order_size, 0)
        self.assertEqual(limit_sell_order2.order_size, 0)

    def test_fill_with_partial_limit_order_multiple_and_price_matching_order(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 195, "LIMIT", -50)
        limit_sell_order2 = Order(ID_GENERATOR, 2, 200, "LIMIT", -25)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order)
        book.add_order(limit_sell_order2)
        self.assertEqual(book.bids, [limit_buy_order])
        self.assertEqual(book.asks, [])
        self.assertEqual(book.bids[0].order_size, 25)
        self.assertEqual(limit_sell_order.order_size, 0)
        self.assertEqual(limit_buy_order.partial_execution_log, [PartialExecution(50, 200), PartialExecution(25, 200)])

    def test_fill_with_partial_limit_order_multiple_and_price_matching_order2(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 200, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 195, "LIMIT", -50)
        limit_sell_order2 = Order(ID_GENERATOR, 2, 200, "LIMIT", -25)
        book.add_order(limit_sell_order)
        book.add_order(limit_buy_order)
        book.add_order(limit_sell_order2)
        self.assertEqual(book.bids, [limit_buy_order])
        self.assertEqual(book.asks, [])
        self.assertEqual(book.bids[0].order_size, 25)
        self.assertEqual(limit_sell_order.order_size, 0)
        self.assertEqual(limit_buy_order.partial_execution_log, [PartialExecution(50, 195), PartialExecution(25, 200)])

    def test_cancelled_orders_in_spread_calc(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 195, "LIMIT", 100)
        limit_sell_order = Order(ID_GENERATOR, 2, 200, "LIMIT", -50)
        book.add_order(limit_sell_order)
        book.add_order(limit_buy_order)
        self.assertEqual(book.current_spread(), 5)
        limit_sell_order.cancel()
        self.assertEqual(book.current_spread(), 0)

    def test_partial_fill_of_market_order(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 195, "LIMIT", 50)
        market_sell_order = Order(ID_GENERATOR, 2, None, "MARKET", -100)
        book.add_order(limit_buy_order)
        book.add_order(market_sell_order)
        self.assertEqual(book.bids, [])
        self.assertEqual(book.asks, [])
        self.assertEqual(market_sell_order.order_state, 'CANCEL_PARTIAL_UNFILLED')
        self.assertEqual(limit_buy_order.order_size, 0)

    def test_partial_fill_of_market_order2(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 195, "LIMIT", 50)
        limit_buy_order2 = Order(ID_GENERATOR, 3, 200, "LIMIT", 50)
        market_sell_order = Order(ID_GENERATOR, 2, None, "MARKET", -100)
        book.add_order(limit_buy_order)
        book.add_order(limit_buy_order2)
        book.add_order(market_sell_order)
        self.assertEqual(book.bids, [])
        self.assertEqual(book.asks, [])
        self.assertEqual(market_sell_order.order_state, 'FILLED')
        self.assertEqual(limit_buy_order.order_size, 0)
        self.assertEqual(limit_buy_order2.order_size, 0)

    def test_partial_fill_of_market_sell_order_existing_unfilled_orders(self):
        book = OrderBook()
        limit_buy_order = Order(ID_GENERATOR, 1, 195, "LIMIT", 50)
        limit_buy_order2 = Order(ID_GENERATOR, 3, 200, "LIMIT", 50)
        limit_buy_order3 = Order(ID_GENERATOR, 4, 180, "LIMIT", 700)
        market_sell_order = Order(ID_GENERATOR, 2, None, "MARKET", -100)
        book.add_order(limit_buy_order)
        book.add_order(limit_buy_order2)
        book.add_order(limit_buy_order3)
        book.add_order(market_sell_order)
        self.assertEqual(book.bids, [limit_buy_order3])
        self.assertEqual(book.asks, [])
        self.assertEqual(market_sell_order.order_state, 'FILLED')
        self.assertEqual(limit_buy_order.order_size, 0)
        self.assertEqual(limit_buy_order2.order_size, 0)
        self.assertEqual(limit_buy_order3.order_size, 700)

    def test_partial_fill_of_market_buy_order_existing_unfilled_orders(self):
        book = OrderBook()
        limit_sell_order = Order(ID_GENERATOR, 1, 195, "LIMIT", -50)
        limit_sell_order2 = Order(ID_GENERATOR, 3, 200, "LIMIT", -50)
        limit_sell_order3 = Order(ID_GENERATOR, 4, 180, "LIMIT", -700)
        market_buy_order = Order(ID_GENERATOR, 2, None, "MARKET", 100)
        book.add_order(limit_sell_order)
        book.add_order(limit_sell_order2)
        book.add_order(limit_sell_order3)
        book.add_order(market_buy_order)
        self.assertEqual(book.bids, [])
        self.assertEqual(book.asks, [limit_sell_order3, limit_sell_order2, limit_sell_order])
        self.assertEqual(market_buy_order.order_state, 'FILLED')
        self.assertEqual(limit_sell_order.order_size, -50)
        self.assertEqual(limit_sell_order2.order_size, -50)
        self.assertEqual(limit_sell_order3.order_size, -600)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
