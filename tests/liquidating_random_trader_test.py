import unittest

from liquidating_random_trader import LiquidatingRandomTrader
from market import Market
from orderbook import next_id, PartialExecution
from trader import Trader, TraderPosition

TRADER_ID = next_id()


class TestLiquidatingRandomTrader(unittest.TestCase):
    def test_should_properly_initialize(self):
        trader = LiquidatingRandomTrader(TRADER_ID, 400, -800)
        self.assertTrue(trader)

    def test_should_trigger_liquidation_on_loss_full_liquidity(self):
        trader = LiquidatingRandomTrader(TRADER_ID, 400, -800)
        # Create a position of 400 @ 400, manually add it to pnl calc
        trader.positions.append(TraderPosition(400, 400))
        trader.pnl_calc.open_trades.append((400, 400))
        # Create two dummies
        dummy_trader = Trader(TRADER_ID, 400)
        dummy_trader_two = Trader(TRADER_ID, 400)
        # Create an order that should move the market spot to 394
        jerk_order = dummy_trader.create_order(TRADER_ID, 394, "LIMIT", -400)
        # Create a buy order under that for 393
        buy_jerk_order = dummy_trader_two.create_order(TRADER_ID, 393, "LIMIT", 400)
        market = Market([dummy_trader, dummy_trader_two, trader])
        market.submit_order(dummy_trader, jerk_order)
        market.submit_order(dummy_trader_two, buy_jerk_order)
        self.assertEqual(market.get_current_spot().best_spot_price(), 394)
        trader.tick(market)
        self.assertEqual(trader.positions[0].price, 400)
        self.assertEqual(trader.positions[0].size, 400)
        self.assertEqual(trader.positions[1].price, 393)
        self.assertEqual(trader.positions[1].size, -400)
        self.assertEqual(trader.unrealized_pnl(market.get_current_spot().best_spot_price()), 0)
        self.assertEqual(trader.realized_pnl(), -2800)
        self.assertEqual(trader.should_continue_to_liquidate, False)

    def test_should_trigger_liquidation_on_loss_partial_liquidity(self):
        trader = LiquidatingRandomTrader(TRADER_ID, 400, -800)
        # Create a position of 400 @ 400, manually add it to pnl calc
        trader.positions.append(TraderPosition(400, 400))
        trader.pnl_calc.open_trades.append((400, 400))
        # Create two dummies
        dummy_trader = Trader(TRADER_ID, 400)
        dummy_trader_two = Trader(TRADER_ID, 400)
        # Create an order that should move the market spot to 394
        jerk_order = dummy_trader.create_order(TRADER_ID, 394, "LIMIT", -400)
        # Create a buy order under that for 393
        buy_jerk_order = dummy_trader_two.create_order(TRADER_ID, 393, "LIMIT", 200)
        market = Market([dummy_trader, dummy_trader_two, trader])
        market.submit_order(dummy_trader, jerk_order)
        market.submit_order(dummy_trader_two, buy_jerk_order)
        self.assertEqual(market.get_current_spot().best_spot_price(), 394)
        trader.tick(market)
        self.assertEqual(trader.positions[0].price, 400)
        self.assertEqual(trader.positions[0].size, 400)
        self.assertEqual(trader.positions[1].price, 393)
        self.assertEqual(trader.positions[1].size, -200)
        self.assertEqual(trader.unrealized_pnl(market.get_current_spot().best_spot_price()), -1200)
        self.assertEqual(trader.realized_pnl(), -1400)
        self.assertEqual(trader.should_continue_to_liquidate, True)
        # Create a buy order under that for 390
        buy_jerk_order = dummy_trader_two.create_order(TRADER_ID, 390, "LIMIT", 200)
        market.submit_order(dummy_trader_two, buy_jerk_order)
        trader.tick(market)
        self.assertEqual(trader.positions[2].price, 390)
        self.assertEqual(trader.positions[2].size, -200)
        self.assertEqual(trader.unrealized_pnl(market.get_current_spot().best_spot_price()), 0)
        self.assertEqual(trader.realized_pnl(), -3400)
        trader.tick(market)
        self.assertEqual(trader.should_continue_to_liquidate, False)

    def test_should_randomly_size_order(self):
        import unittest.mock as mock

        def mock_random(c, p):
            return c[-2]

        with mock.patch('numpy.random.choice', mock_random):
            trader = LiquidatingRandomTrader(TRADER_ID, 400, -800)
            dummy_trader = Trader(TRADER_ID, 400)
            market = Market([dummy_trader, trader])
            jerk_order = dummy_trader.create_order(TRADER_ID, 394, "LIMIT", -200)
            market.submit_order(dummy_trader, jerk_order)
            trader.tick(market)
            self.assertEqual(jerk_order.partial_execution_log[0], PartialExecution(-100, 394))
            self.assertEqual(jerk_order.order_state, "PARTIAL_EXECUTION")
            trader.tick(market)
            self.assertEqual(jerk_order.partial_execution_log[1], PartialExecution(-100, 394))
            self.assertEqual(jerk_order.order_state, "FILLED")


    def test_should_never_exceed_max_size(self):
        import unittest.mock as mock

        def mock_random(c, p):
            return 400

        with mock.patch('numpy.random.choice', mock_random):
            trader = LiquidatingRandomTrader(TRADER_ID, 400, -800)
            dummy_trader = Trader(TRADER_ID, 400)
            market = Market([dummy_trader, trader])
            jerk_order = dummy_trader.create_order(TRADER_ID, 394, "LIMIT", -500)
            market.submit_order(dummy_trader, jerk_order)
            trader.tick(market)
            self.assertEqual(jerk_order.partial_execution_log[0], PartialExecution(-400, 394))
            self.assertEqual(jerk_order.order_state, "PARTIAL_EXECUTION")
            trader.tick(market)
            self.assertEqual(len(jerk_order.partial_execution_log), 1)
            self.assertEqual(jerk_order.order_state, "PARTIAL_EXECUTION")
