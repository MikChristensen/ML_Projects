import numpy as np
import pandas as pd
from enum import Enum

class TimeFrame(Enum):
  DAILY = 252
  MIN_1 = 252 * 24 * 60
  MIN_5 = 252 * 24 * 12
  MIN_10 = 252 * 24 * 6
  HOUR_1 = 252 * 24
  HOUR_6 = 252 * 4

class Backtest:
    def __init__(self):

        selected_timeframe = TimeFrame.MIN_1

        pass

    def compute_beta(self, return_serie, market_serie):
        # We need compute the covariance between the market and the portfolio 
        val = pd.concat((return_serie, market_serie), axis=1).dropna()
        
        # We compute beta 
        # We compute beta 
        cov_var_mat = np.cov(val.values, rowvar=False)
        cov = cov_var_mat[0][1]
        var = cov_var_mat[1][1]

        beta = cov/var

        return beta
    
    def compute_alpha(self, return_serie, market_serie):
        # We need compute the covariance between the market and the portfolio 
        val = pd.concat((return_serie, market_serie), axis=1).dropna()
        
        # We compute beta 
        cov_var_mat = np.cov(val.values, rowvar=False)
        beta = cov_var_mat[0, 1] / cov_var_mat[1, 1]
        
        # We compute alpha
        alpha = val.iloc[:, 0].mean() - beta * val.iloc[:, 1].mean()
        return alpha
    
    def compute_sharpe_ratio(self, return_serie, risk_free_rate=0.0):
        excess_return = return_serie - risk_free_rate
        sharpe_ratio = excess_return.mean() / excess_return.std()
        return sharpe_ratio
    
    def compute_drawdown(self, return_serie):
        cumulative_return = (1 + return_serie).cumprod()
        running_max = cumulative_return.cummax()
        drawdown = (cumulative_return - running_max) / running_max
        return drawdown.min()

