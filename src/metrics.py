"""
metrics.py

Utility functions for evaluating trading strategy performance.

This module contains commonly used risk-adjusted performance metrics such as:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

All functions are designed to work with pandas Series objects.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, freq: int = 252 * 78) -> float:
    """
    Compute the annualized Sharpe Ratio of a return series.

    Parameters
    ----------
    returns : pd.Series
        Strategy return series (periodic returns).
    freq : int, optional
        Annualization factor. Default corresponds to 5-minute bars:
        252 trading days * ~78 bars per day.

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0 or np.isnan(std_ret):
        return 0.0

    return (mean_ret / std_ret) * np.sqrt(freq)


def sortino_ratio(returns: pd.Series, freq: int = 252 * 78) -> float:
    """
    Compute the annualized Sortino Ratio of a return series.

    Sortino ratio penalizes only downside volatility.

    Parameters
    ----------
    returns : pd.Series
        Strategy return series (periodic returns).
    freq : int, optional
        Annualization factor. Default corresponds to 5-minute bars:
        252 trading days * ~78 bars per day.

    Returns
    -------
    float
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return (returns.mean() / downside_std) * np.sqrt(freq)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (e.g., cumulative returns or portfolio value).

    Returns
    -------
    float
        Maximum drawdown as a negative float (e.g., -0.25 = -25%).
    """
    equity_curve = equity_curve.dropna()

    if len(equity_curve) == 0:
        return 0.0

    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1.0

    return drawdown.min()


def calmar_ratio(equity_curve: pd.Series, returns: pd.Series) -> float:
    """
    Compute the Calmar Ratio of a trading strategy.

    Calmar Ratio = Total Return / |Maximum Drawdown|

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.
    returns : pd.Series
        Periodic returns of the strategy.

    Returns
    -------
    float
        Calmar ratio. Returns 0.0 if maximum drawdown is zero.
    """
    equity_curve = equity_curve.dropna()
    returns = returns.dropna()

    if len(equity_curve) == 0:
        return 0.0

    mdd = abs(max_drawdown(equity_curve))

    if mdd == 0 or np.isnan(mdd):
        return 0.0

    total_return = equity_curve.iloc[-1] - 1.0

    return total_return / mdd
