import unittest

from market import Market
from market_making_trader import MarketMakingTrader
from orderbook import next_id
from trader import TraderPosition, Trader

TRADER_ID = next_id()
ORDER_ID = next_id()


class TestMarketMakingTrader(unittest.TestCase):
    def test_should_properly_initialize(self):
        self.assertTrue(MarketMakingTrader(TRADER_ID, 400))

    def test_should_kill_all_active_orders_on_tick(self):
        trader = MarketMakingTrader(TRADER_ID, 400)
        market = Market([trader])
        order = trader.create_order(ORDER_ID, 400, "LIMIT", 400)
        trader.orders.append(order)
        trader.tick(market)
        self.assertEqual(order.order_state, "CANCELLED")

    def test_should_create_spread_properly(self):
        trader = MarketMakingTrader(TRADER_ID, 400)
        market = Market([trader])
        trader.tick(market)
        self.assertEqual(len(market.orderbook.bids), 1)
        self.assertEqual(len(market.orderbook.asks), 1)
        self.assertEqual(len(trader.orders), 2)
        self.assertEqual(list(map(lambda x: x.price, trader.orders)), [0.02, 0.01])

    def test_should_liquidate_position(self):
        trader = MarketMakingTrader(TRADER_ID, 400)
        market = Market([trader])
        trader.positions.append(TraderPosition(400, 400))
        trader.tick(market)
        self.assertEqual(len(trader.orders), 1)
        self.assertEqual(trader.orders[0].price, 0.02)
        self.assertEqual(trader.orders[0].order_size, -400)

    def test_should_create_two_sided_spread_otherwise(self):
        trader = MarketMakingTrader(TRADER_ID, 400)
        dummy_trader = Trader(TRADER_ID, 400)
        market = Market([trader])
        market.submit_order(dummy_trader, dummy_trader.create_order(ORDER_ID, 200, "LIMIT", 200))
        trader.tick(market)
        self.assertEqual(len(trader.orders), 2)
        self.assertEqual(trader.orders[1].price, 200)
        self.assertEqual(trader.orders[1].order_size, 400)
        self.assertEqual(trader.orders[0].price, 200.01)
        self.assertEqual(trader.orders[0].order_size, -400)
