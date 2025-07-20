import pandas as pd

def simple_commit_threshold_strategy(commit_series, buy_threshold, sell_threshold):
    """
    Simple trading strategy based on Commit count thresholds.
    Buys when Commit count exceeds buy threshold, sells when below sell threshold.
    Assumes full position buy/sell.
    """
    signals = pd.Series(0, index=commit_series.index, dtype=int)
    holding = 0 # 0: no position, 1: holding position

    if commit_series.empty:
        return signals

    for i in range(len(commit_series)):
        current_date = commit_series.index[i]
        current_commits = commit_series.iloc[i]

        if holding == 0: # No position
            if current_commits >= buy_threshold:
                signals.iloc[i] = 1 # Buy signal
                holding = 1
            else:
                signals.iloc[i] = 0 # Remain no position
        elif holding == 1: # Holding position
            if current_commits <= sell_threshold:
                signals.iloc[i] = -1 # Sell signal
                holding = 0
            else:
                signals.iloc[i] = 0 # Continue holding
    
    return signals
