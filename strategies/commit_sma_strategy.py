import pandas as pd

def commit_sma_strategy(commit_series, short_period, long_period):
    """
    Strategy based on Commit count SMA golden cross/death cross.
    Golden cross (short-period SMA crosses above long-period SMA) triggers buy.
    Death cross (short-period SMA crosses below long-period SMA) triggers sell.
    """
    signals = pd.Series(0, index=commit_series.index, dtype=int)
    holding = 0 # 0: no position, 1: holding position

    if commit_series.empty or len(commit_series) < long_period:
        return signals # Not enough data to calculate SMA

    short_sma = commit_series.rolling(window=short_period, min_periods=1).mean()
    long_sma = commit_series.rolling(window=long_period, min_periods=1).mean()

    for i in range(1, len(commit_series)):
        # Ensure SMA values are not NaN
        if pd.isna(short_sma.iloc[i]) or pd.isna(long_sma.iloc[i]):
            continue

        # Golden Cross: Short-period SMA crosses above long-period SMA
        if short_sma.iloc[i-1] <= long_sma.iloc[i-1] and short_sma.iloc[i] > long_sma.iloc[i] and holding == 0:
            signals.iloc[i] = 1 # Buy signal
            holding = 1
        # Death Cross: Short-period SMA crosses below long-period SMA
        elif short_sma.iloc[i-1] >= long_sma.iloc[i-1] and short_sma.iloc[i] < long_sma.iloc[i] and holding == 1:
            signals.iloc[i] = -1 # Sell signal
            holding = 0
        else:
            signals.iloc[i] = 0 # No action
    
    return signals
