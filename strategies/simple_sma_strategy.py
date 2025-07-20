import pandas as pd

def simple_sma_strategy(price_series, sma1_period, sma2_period):
    """
    Generates buy/sell signals based on a simple SMA crossover strategy.
    Buy when SMA1 (shorter period) crosses above SMA2 (longer period) (Golden Cross).
    Sell when SMA1 (shorter period) crosses below SMA2 (longer period) (Death Cross).

    Args:
        price_series (pd.Series): A pandas Series of daily closing prices, indexed by date.
        sma1_period (int): The period for the shorter Simple Moving Average.
        sma2_period (int): The period for the longer Simple Moving Average.

    Returns:
        pd.Series: A pandas Series of trading signals (1 for buy, -1 for sell, 0 for hold), indexed by date.
    """
    if sma1_period >= sma2_period:
        raise ValueError("SMA1 period must be shorter than SMA2 period.")

    # Calculate SMAs
    sma1 = price_series.rolling(window=sma1_period, min_periods=1).mean()
    sma2 = price_series.rolling(window=sma2_period, min_periods=1).mean()

    # Initialize signals
    signals = pd.Series(0, index=price_series.index, dtype=int)

    # Generate signals based on crossovers
    # A crossover happens when the difference changes sign
    # Golden Cross: sma1 crosses above sma2 (sma1 - sma2 changes from negative to positive)
    # Death Cross: sma1 crosses below sma2 (sma1 - sma2 changes from positive to negative)
    
    # Calculate the difference between SMA1 and SMA2
    sma_diff = sma1 - sma2

    # Find where the difference crosses zero
    # Shifted difference helps detect the crossover point
    golden_cross = (sma_diff.shift(1) < 0) & (sma_diff >= 0)
    death_cross = (sma_diff.shift(1) > 0) & (sma_diff <= 0)

    signals[golden_cross] = 1  # Buy signal
    signals[death_cross] = -1 # Sell signal

    # Ensure no signals are generated for the initial period where SMAs are not fully formed
    # Or where sma1_period is not met for sma1, or sma2_period for sma2
    signals[:sma2_period-1] = 0 # No signals until both SMAs have enough data

    return signals
