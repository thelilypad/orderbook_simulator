from market import Market
from orderbook import Order

'''
A simple wrapper class for keeping the position lists (both closed and open) of a given trader
'''


class TraderPosition:
    def __init__(self, size: int, price: float):
        self.size = size
        self.price = price

    '''
    Returns the notional value of the position (the size * price)
    '''

    def notional(self) -> float:
        return self.size * self.price


'''
Helper class for FIFO calculation of a trader's PNL.
It handles both maintaining a running tally of PnL as well as both unrealized and realized PNL calculations.
'''

class PnlCalculator:
    def __init__(self):
        self.open_trades = []
        self.closed_trades = []

    '''
    Returns the realized (closed) PnL
    '''
    def realized_pnl(self) -> float:
        return sum([t[0] * t[1] for t in self.closed_trades])

    '''
    Returns unrealized (open) PnL
    :param current_asset_price: the current spot price
    '''
    def unrealized_pnl(self, current_asset_price) -> float:
        return sum([t[0] * (current_asset_price - t[1]) for t in self.open_trades])

    def fill(self, n_pos, exec_price):
        current_quantity = sum([t[0] for t in self.open_trades])
        if not self.open_trades:
            self.open_trades.append((n_pos, exec_price))
            return
        # Assuming the same sign of the new quantity to add
        if n_pos * current_quantity > 0:
            self.open_trades.append((n_pos, exec_price))
            # Case 1: Receiving new fills that increase your position
        # Assuming a different sign of the new quantity to add
        elif n_pos * current_quantity < 0:
            # Case 2: Receiving new fills that decrease your position
            total_to_fill = 0
            # Pluck first in first out positions according to cost basis until we cover
            # the whole order
            while abs(total_to_fill) <= abs(n_pos) and self.open_trades:
                # Pop the first-in open position (front of the list)
                size, cost = self.open_trades.pop(0)
                # Calculate the remainder we need to fill from the position just added
                # and the total we've filled so far
                total_remaining_to_fill = (n_pos + total_to_fill) * -1
                if abs(size) <= abs(total_remaining_to_fill):
                    self.closed_trades.append((size, exec_price - cost))
                else:
                    self.open_trades = [(size - total_remaining_to_fill, cost)] + self.open_trades
                    self.closed_trades.append((total_remaining_to_fill, exec_price - cost))
                total_to_fill = total_to_fill + size

            # Case 3: Reverse your position
            if abs(total_to_fill) < abs(n_pos):
                remaining_fill = n_pos + total_to_fill
                self.open_trades.append((remaining_fill, exec_price))


'''
Base class for all Trader/Trader types, implementing basic trader functionality (add/close positions, PnL calculation, current_position size).
Additionally all super-classes must implement the method tick(), which will be called on every "turn" of the market.
'''


class Trader:
    def __init__(self, id_generator, max_account_size: int, positions: list = []):
        self.positions = positions
        self.max_account_size = max_account_size
        self.agent_id = next(id_generator)
        self.pnl_calc = PnlCalculator()

    '''
    Default handling method for creating an order which will actually change a trader's active positions.
    :param id_generator
    
    '''

    def create_order(self, id_generator, price: float, order_type: str, order_size: int) -> Order:
        return Order(id_generator, self.agent_id, price, order_type, order_size, self.handle_order_fill)

    '''
    Default handling method for when an order is filled (partially). We should modify the trader's positions and cost basis.
    '''

    def handle_order_fill(self, order: Order):
        if not order.partial_execution_log:
            raise AssertionError("Should not trigger order fill handler without order fill!")
        last_executed = order.partial_execution_log[-1]
        self.add_units_at_price(last_executed.quantity, last_executed.at_price)

    '''
    Adds a given position to the trader
    '''

    def add_units_at_price(self, size: int, price: float):
        self.positions.append(TraderPosition(size, price))
        self.pnl_calc.fill(size, price)

    '''
    Returns the realized PnL of the trader
    '''

    def realized_pnl(self) -> float:
        return self.pnl_calc.realized_pnl()

    '''
    Returns the unrealized PnL of the trader
    :param current_asset_price: the current spot price of the asset to calc unrealized PnL
    '''

    def unrealized_pnl(self, current_asset_price: float) -> float:
        return self.pnl_calc.unrealized_pnl(current_asset_price)

    '''
    Returns the total amount of units of the asset the trader currently has
    '''

    def current_position(self) -> int:
        return sum([t.size for t in self.positions])

    '''
    Implements the "turn" of the trader in the market sim.
    
    :param market: (instance of Market)
    '''

    def tick(self, market: Market):
        raise NotImplementedError("All traders must implement the tick method!")
