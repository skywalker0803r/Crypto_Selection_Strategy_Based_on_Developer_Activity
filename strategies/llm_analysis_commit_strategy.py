import pandas as pd
import time
from llm_analysis_commit import get_llm_analysis, load_cache, save_cache

def llm_strategy_generator(commit_df, price_series, buy_score_threshold, sell_score_threshold, 
                           initial_capital, commission_rate, currency, progress=None, return_signals_only=False):
    """
    A generator that analyzes commits day-by-day, generates signals, runs the backtest incrementally,
    and yields the state of the portfolio for each processed day.
    """
    # --- Initialization from backtester ---
    portfolio_value = pd.Series(index=price_series.index, dtype='float64')
    current_cash = initial_capital
    holding_shares = 0
    buy_points = []
    sell_points = []
    trades_info = []
    open_trade = None
    
    # For returning signals only
    signals_series = pd.Series(0, index=price_series.index, dtype=int)

    # --- Cache Optimization: Load cache once and process new commits in batch ---
    cache = load_cache()
    all_commit_messages = commit_df['message'].unique()
    new_commit_messages = [msg for msg in all_commit_messages if msg not in cache]

    if new_commit_messages:
        print(f"Found {len(new_commit_messages)} new commit messages to analyze.")
        for idx, message in enumerate(new_commit_messages):
            if progress:
                progress(idx / len(new_commit_messages), desc=f"Analyzing new commits with LLM ({idx+1}/{len(new_commit_messages)})")
            get_llm_analysis(message, cache) # get_llm_analysis now updates the cache directly
        save_cache(cache) # Save cache after all new commits are analyzed
        print("Finished analyzing new commit messages and updated cache.")
    else:
        print("No new commit messages to analyze. Using existing cache.")

    # --- Main Loop: Iterate through each day in the price series ---
    print("DEBUG: Starting main loop through price series.")
    total_days = len(price_series)
    
    for i, (current_date, current_price) in enumerate(price_series.items()):
        if progress:
            progress(i / total_days, desc=f"Backtesting Day {i + 1}/{total_days}")

        # 1. Analyze commits for the current day using pre-analyzed results
        todays_commits = commit_df[commit_df['date'].dt.date == current_date.date()]
        daily_score = 0
        if not todays_commits.empty:
            for _, row_data in enumerate(todays_commits.iterrows()):
                commit_message = row_data[1]['message']
                analysis_result = cache.get(commit_message, {"對幣價的影響": "未知"}) # Get from cache
                impact = analysis_result.get("對幣價的影響", "無明顯影響")
                if "上漲" in impact:
                    daily_score += 1
                elif "下跌" in impact:
                    daily_score -= 1

        # 2. Generate signal for the current day
        signal = 0
        if holding_shares == 0: # If not holding, check for buy signal
            if daily_score >= buy_score_threshold:
                signal = 1
        else: # If holding, check for sell signal
            if daily_score <= sell_score_threshold:
                signal = -1

        signals_series.loc[current_date] = signal

        if return_signals_only:
            # If only signals are requested, continue to next day without backtesting logic
            continue

        # 3. Update portfolio based on the signal (logic from backtester)
        if pd.isna(current_price):
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] if i > 0 else initial_capital
        else:
            if signal == 1 and open_trade is None and current_cash > 0:
                shares_to_buy = (current_cash / current_price) * (1 - commission_rate)
                holding_shares = shares_to_buy
                current_cash = 0
                open_trade = {'buy_date': current_date, 'buy_price': current_price, 'shares': holding_shares}
                buy_points.append({'date': current_date, 'price': current_price})

            elif signal == -1 and open_trade is not None:
                revenue = open_trade['shares'] * current_price * (1 - commission_rate)
                commission_cost = open_trade['shares'] * current_price * commission_rate + open_trade['buy_price'] * open_trade['shares'] * commission_rate
                profit_loss = revenue - (open_trade['shares'] * open_trade['buy_price'])
                
                # Calculate profit percentage
                profit_percentage = (profit_loss / (open_trade['shares'] * open_trade['buy_price'])) * 100 if (open_trade['shares'] * open_trade['buy_price']) != 0 else 0

                # Calculate current total capital and return percentage at the time of trade
                current_total_capital = current_cash + (holding_shares * current_price) # This should be the portfolio value *after* the trade
                current_total_return_percentage = (current_total_capital / initial_capital - 1) * 100

                trades_info.append({
                    '時間': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    '幣種': currency.upper(),
                    '買進價格': open_trade['buy_price'],
                    '賣出價格': current_price,
                    '買進時間': open_trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    '賣出時間': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    '持有時間': (current_date - open_trade['buy_date']).days,
                    '利潤': profit_loss,
                    '利潤%數': profit_percentage,
                    '當前總資產': current_total_capital,
                    '當前總報酬率%數': current_total_return_percentage,
                    'commission_cost': commission_cost # Re-add commission_cost
                })
                current_cash = revenue
                holding_shares = 0
                open_trade = None
                sell_points.append({'date': current_date, 'price': current_price})
            
            portfolio_value.iloc[i] = current_cash + (holding_shares * current_price)

        # 4. Yield the current state (logic from backtester)
        portfolio_snapshot = portfolio_value.iloc[:i+1].dropna()
        if portfolio_snapshot.empty:
            continue
            
        cumulative_returns = portfolio_snapshot / initial_capital
        
        # Calculate Buy and Hold for the same period
        buy_and_hold_snapshot = price_series.iloc[:i+1].dropna()
        if not buy_and_hold_snapshot.empty and buy_and_hold_snapshot.iloc[0] > 0:
            buy_and_hold_returns = buy_and_hold_snapshot / buy_and_hold_snapshot.iloc[0]
        else:
            buy_and_hold_returns = pd.Series(1.0, index=cumulative_returns.index)

        # Calculate Max Drawdown
        if not cumulative_returns.empty:
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0.0

        # Calculate Trade Statistics
        total_trades = len(trades_info)
        winning_trades = [t for t in trades_info if t['利潤'] > 0]
        gross_profit = sum(t['利潤'] for t in winning_trades)
        losing_trades = [t for t in trades_info if t['利潤'] < 0]
        gross_loss = sum(t['利潤'] for t in losing_trades)
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        total_commission_paid = sum(t['commission_cost'] for t in trades_info)

        # Calculate new metrics
        average_holding_days = sum(t['持有時間'] for t in trades_info) / total_trades if total_trades > 0 else 0
        average_profit_per_trade = sum(t['利潤'] for t in trades_info) / total_trades if total_trades > 0 else 0
        trade_frequency = total_trades / total_days # Trades per day

        performance_metrics = {
            "Status": f"Processing day {i+1}/{total_days}",
            "Selected Strategy Total Return (with commissions)": f"{(cumulative_returns.iloc[-1] - 1) * 100:.2f}%",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Buy and Hold Strategy Total Return": f"{(buy_and_hold_returns.iloc[-1] - 1) * 100:.2f}%",
            "Final Capital (with commissions)": f"{portfolio_snapshot.iloc[-1]:.2f} {currency.upper()}",
            "--- Trade Statistics ---": "",
            "Total Trades": total_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinity",
            "Total Commission Paid": f"{total_commission_paid:.2f} {currency.upper()}",
            "Average Holding Days": f"{average_holding_days:.2f} days",
            "Average Profit per Trade": f"{average_profit_per_trade:.2f} {currency.upper()}",
            "Trade Frequency (per day)": f"{trade_frequency:.4f}"
        }
        
        yield cumulative_returns, buy_and_hold_returns, buy_points, sell_points, performance_metrics, trades_info
    
    print("DEBUG: Main loop finished.")
    if return_signals_only:
        print("DEBUG: Yielding signals_series as return_signals_only is True.")
        yield None, None, None, None, None, signals_series