import os, sys
import unittest

from orderbook import next_id

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trader import TraderPosition, PnlCalculator, Trader


class TestTraderPosition(unittest.TestCase):
    def test_calculates_correct_notional(self):
        pos = TraderPosition(50, 50.3)
        self.assertEqual(pos.notional(), 50 * 50.3)

class TestPnlCalculator(unittest.TestCase):
    def test_random_stack_overflow_case(self):
        # Sanity check case from:
        # https://quant.stackexchange.com/questions/9002/calculate-average-price-cost-unrealized-pl-of-a-position-based-on-executed
        calc = PnlCalculator()
        calc.fill(1, 80.0)
        calc.fill(-3, 102.0)
        calc.fill(-2, 98.0)
        calc.fill(3, 90.0)
        calc.fill(-2, 100.0)
        self.assertEqual(calc.realized_pnl(), 54)

    def test_calc_pnl_fifo_long_initial(self):
        calc = PnlCalculator()
        # Add one position
        calc.fill(50, 50)
        self.assertEqual(calc.unrealized_pnl(60), 500)
        self.assertEqual(calc.unrealized_pnl(70), 1000)
        self.assertEqual(calc.realized_pnl(), 0)
        # Close that position
        calc.fill(-50, 70)
        self.assertEqual(calc.unrealized_pnl(70), 0)
        self.assertEqual(calc.realized_pnl(), 1000)
        # Reopen a new position at two different cost bases
        calc.fill(20, 65)
        calc.fill(20, 70)
        self.assertEqual(calc.unrealized_pnl(70), 100)
        self.assertEqual(calc.unrealized_pnl(75), 300)
        self.assertEqual(calc.realized_pnl(), 1000)
        # Close part of the position
        calc.fill(-30, 75)
        self.assertEqual(calc.unrealized_pnl(75), 50)
        self.assertEqual(calc.realized_pnl(), 1250)
        # Take a loss on the remaining part
        self.assertEqual(calc.unrealized_pnl(50), -200)
        calc.fill(-10, 50)
        self.assertEqual(calc.unrealized_pnl(50), 0)
        self.assertEqual(calc.realized_pnl(), 1050)

    def test_calc_pnl_fifo_short_initial(self):
        calc = PnlCalculator()
        # Add one position
        calc.fill(-50, 50)
        self.assertEqual(calc.unrealized_pnl(60), -500)
        self.assertEqual(calc.unrealized_pnl(70), -1000)
        self.assertEqual(calc.realized_pnl(), 0)
        # Close that position
        calc.fill(50, 70)
        self.assertEqual(calc.unrealized_pnl(70), 0)
        self.assertEqual(calc.realized_pnl(), -1000)
        # Reopen a new position at two different cost bases
        calc.fill(-20, 65)
        calc.fill(-20, 70)
        self.assertEqual(calc.unrealized_pnl(70), -100)
        self.assertEqual(calc.unrealized_pnl(75), -300)
        self.assertEqual(calc.realized_pnl(), -1000)
        # Close part of the position
        calc.fill(30, 75)
        self.assertEqual(calc.unrealized_pnl(75), -50)
        self.assertEqual(calc.realized_pnl(), -1250)
        # Take a loss on the remaining part
        self.assertEqual(calc.unrealized_pnl(50), 200)
        calc.fill(10, 50)
        self.assertEqual(calc.unrealized_pnl(50), 0)
        self.assertEqual(calc.realized_pnl(), -1050)

TRADER_ID = next_id()
ORDER_ID = next_id()

class TestTrader(unittest.TestCase):
    def test_create_order_correctly(self):
        trader = Trader(TRADER_ID, 400)
        order = trader.create_order(ORDER_ID, 100, 'LIMIT', 100)
        self.assertEqual(order.order_type, "LIMIT")
        self.assertEqual(order.order_size, 100)
        self.assertEqual(order.price, 100)
        self.assertEqual(order.agent_id, trader.agent_id)
        self.assertRaises(AssertionError, order.order_fill_callback, order)

    def test_order_fill_handler(self):
        trader = Trader(TRADER_ID, 400)
        order = trader.create_order(ORDER_ID, 100, 'LIMIT', 100)
        self.assertRaises(AssertionError, trader.handle_order_fill, order)
        order.partial_execute(40, 100)
        trader.handle_order_fill(order)
        self.assertEqual(trader.positions[0].__dict__, {'price': 100, 'size': 40})


    def test_add_units_and_update_pnl(self):
        trader = Trader(TRADER_ID, 400)
        trader.add_units_at_price(40, 50)
        self.assertEqual(trader.positions[0].__dict__, {'price': 50, 'size': 40})
        trader.add_units_at_price(-40, 60)
        self.assertEqual(trader.realized_pnl(), 400)

