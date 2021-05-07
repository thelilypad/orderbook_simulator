'''
A simple wrapper class for keeping the position lists (both closed and open) of a given trader
'''
class TraderPosition:
    def __init__(self, size, price):
        self.size = size
        self.price = price
        
    '''
    Returns the notional value of the position (the size * price)
    '''
    def notional():
        return self.size * self.price

'''
Helper class for LIFO or FIFO calculation of a trader's PNL.
It handles both maintaining a running tally of PnL as well as both unrealized and realized PNL calculations.
'''
class PnlCalculator:
    def __init__(self, isFifo=True):
        self.open_trades = []
        self.closed_trades = []
        # By default we should use FIFO accounting principles, but this can be changed if needed to LIFO
        self.fifoIndex = -1 if isFifo else 0
        self.r_pnl = 0
        self.quantity = 0
        self.average_price = 0

    def pnl(self,d):
        return (d['close_price']-d['open_price'])*d['pos']

    '''
    Handles addition to running tally of realized and unrealized PNL.
    :param pos_change (integer): the quantity (in units) added (or subtracted) to the trader's position
    :param exec_price (float): the price the trader is executed at
    '''
    def update_pnl(self, pos_change, exec_price):
        
        def handle_position_closing(self, pos_change, exec_price, last_open):
            # We can add an entry to our closed trades representing the price we long/shorted at (open price) and the closing price (the price we passed in)
            d = {'pos':-pos_change, 'open_price':last_open['price'], 'close_price':exec_price}
            self.closed_trades += [d]
            # Since the trade is closed, we can modify our realized PNL
            self.r_pnl += self.pnl(d)

        
        # If the trader has no positions open, instantiate the new trades array
        if not len(self.open_trades):
            self.open_trades = [{'pos':pos_change, 'price':exec_price}]
            return
        # Fetch the last trade (-1 if FIFO index, 0 otherwise)
        last_open = self.open_trades[self.fifoIndex]
        # If the last trade total and this trade both share the same sign (e.g. the position net is short and we add to our short, or vice versa)
        if last_open['pos']*pos_change>0:
            # Add another entry to our open trades
            self.open_trades += [{'pos':pos_change, 'price':exec_price}]
            return
        # If the size of our last open position is greater or equal to the current change we're adding
        if abs(last_open['pos'])>=abs(pos_change):
            handle_position_closing(self, pos_change, exec_price, last_open)
            # Finally, let's modify the last open position size
            last_open['pos'] += pos_change
            if last_open['pos']==0:
                # And if that position now has 0 units, remove it from the open trades
                self.open_trades.pop(self.fifoIndex)
            return
        # Finally, if this trade is bigger than the last open position
        handle_position_closing(self, pos_change, exec_price, last_open)
        pos_change += last_open['pos']
        self.open_trades.pop(self.fifoIndex)
        self.update_pnl(pos_change, exec_price)

    # This takes in the current asset spot price and our existing positions (as provided in the #fill method) and gives us the unrealized PnL
    def update_unrealized_pnl(self, price):
        u_pnl = 0
        self.quantity = 0
        self.average_price = 0
        for r in self.open_trades:
            u_pnl += r['pos']*(price-r['price'])
            self.quantity += r['pos']
            self.average_price += r['pos']*r['price']
        if self.quantity!=0:
            self.average_price /= self.quantity
        return u_pnl
    
'''
Base class for all Trader/Trader types, implementing basic trader functionality (add/close positions, PnL calculation, current_position size).
Additionally all super-classes must implement the method tick(), which will be called on every "turn" of the market.
'''
class Trader:
    def __init__(self, id_generator,  max_account_size, positions = []):
        self.positions = positions
        self.max_account_size = max_account_size
        self.agent_id = next(id_generator)
        self.pnl_calc = PnlCalculator()
    
    '''
    Adds a given position to the trader
    '''
    def add_units_at_price(self, size, price):
        self.positions.append(TraderPosition(size, price))
        self.pnl_calc.update_pnl(price, size)
    
    '''
    Returns the realized PnL of the trader
    '''
    def realized_pnl(self):
        return self.pnl_calc.r_pnl
    
    '''
    Returns the unrealized PnL of the trader
    :param current_asset_price: the current spot price of the asset to calc unrealized PnL
    '''
    def unrealized_pnl(self, current_asset_price):
        return self.pnl_calc.update_unrealized_pnl(current_asset_price)
    
    '''
    Returns the total amount of units of the asset the trader currently has
    '''
    def current_position(self):
        return sum([t.size for t in self.positions])
    
    '''
    Implements the "turn" of the trader in the market sim.
    
    :param market: (instance of Market)
    '''
    def tick(market):
        raise NotImplementedError("All traders must implement the tick method!")
    

