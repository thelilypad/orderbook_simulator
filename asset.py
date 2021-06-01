import math
import random
import numpy as np
def NUMBER_OF_DAYS_PER_YEAR = 252


class AffineJumpDiffusionModel:
    class Results:
        def __init__(self, returns, vols, jumps):
            self.returns = returns
            self.vols = vols
            self.jumps = jumps

        def sharpe(self):
            pass

    """
    Implementation of the Heston model of stochastic volatility for simulating returns.
    :param starting_price: the current stock price
    :param starting_volatility: the
    :rate_of_return: the drift constant, annualized
    :kappa: The rate at which volatility reverts back to theta (the historic volatility average)
    :theta: The historic average volatility (the long variance)
    :vol_of_vol: The volatility of volatility (the variance of V_t)
    :rho: The correlation between the random (Wiener) processes of volatility and the asset price.
    In general for equities we expect rho to be negative (spot-vol correlation)
    """
    def __init__(self, starting_price, starting_volatility, rate_of_return, kappa, theta, vol_of_vol, rho, mean_jump_size, stdev_jump, rate_of_jumps):
        self.current_price = starting_price
        self.current_volatility = starting_volatility
        self.kappa = kappa
        self.theta = theta
        self.vol_of_vol = vol_of_vol
        self.rho = rho
        self.mean_jump_size = mean_jump_size
        self.stdev_jump = stdev_jump
        self.daily_rate_of_return = rate_of_return ** (1/NUMBER_OF_DAYS_PER_YEAR)
        self.daily_rate_of_jumps = rate_of_jumps/NUMBER_OF_DAYS_PER_YEAR

    def get_d_spot(self):
        # Create a simple zero matrix representing the means of both the price and volatility distributions
        mu = np.array([0, 0])
        # Map our correlation to a covariance matrix of the form
        # (1 rho)
        # (rho 1)
        cov = np.matrix([[1, self.rho][self.rho, 1]])
        # From our mu and cov, we can assume a normal distribution of our increments for both the volatility and spot
        # random processes. What this does is creates two normal random variables which are plucked from a shared
        # distribution, with the level of correlation between them equaling self.rho
        wiener_step = np.random.multivariate_normal(mu, cov, 1)
        # Wiener independent increment of the stock price process
        wiener_s =  wiener_step[:, 0]
        # Wiener independent increment of the volatility process
        wiener_v = wiener_step[:, 1]
        # Given our parameters, we can use the Heston model to calculate dV_t, the derivative/change at each timestep of the volatility process
        d_vol = self.kappa*(self.theta - self.current_volatility) + self.vol_of_vol*math.sqrt(self.current_volatility)*wiener_v
        # We can then modify the volatility accordingly
        self.current_volatility = self.current_volatility + d_vol
        # Finally, we can calculate dS_t, the change at each timestep of the spot price
        d_spot = self.daily_rate_of_return*self.current_price + math.sqrt(self.current_volatility)*self.current_price*wiener_s
        self.current_price = self.current_price + d_spot
        # Next, we should calculate if a jump should occur according to our Poisson process
        # By the definition of the Poisson process, we have a constant probability at each timestep of a jump according
        # which is the expected rate of jumps per day (given the yearly rate)
        # We can select from a uniform distribution a random number between 0 and 1, and if number <= expected rate
        # we can simulate a jump.
        proba = random.uniform(0, 1)
        # We should select from a lognormal distribution (with mean mean_jump_size and stdev stdev_jump) to get the jump magnitude
        jump_size = self.current_price * random.lognormvariate(self.mean_jump_size, self.stdev_jump) if proba <= self.daily_rate_of_jumps else 0
        self.current_price = self.current_price + jump_size
        return d_vol, d_spot, jump_size

    def simulate_results(self, iterations = 50000):
        vols, spots, jumps = map(list, zip(*[self.get_d_spot() for i in range(iterations)]))
        return AffineJumpDiffusionModel.Results(spots, vols, jumps)

