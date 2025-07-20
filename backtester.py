import pandas as pd

def run_backtest(price_series, signals, initial_capital, commission_rate, currency):
    """
    Performs a simple backtest as a generator, yielding results for each day.
    This allows for real-time updates in the UI.
    """
    # --- Initialization ---
    portfolio_value_with_commission = pd.Series(index=price_series.index, dtype='float64')
    portfolio_value_no_commission = pd.Series(index=price_series.index, dtype='float64')
    
    current_cash_wc = initial_capital
    holding_shares_wc = 0 
    current_cash_nc = initial_capital
    holding_shares_nc = 0

    buy_points = []
    sell_points = []
    trades_info = []
    open_trade = None

    if price_series.empty or signals.empty:
        # Yield a single error state if data is missing
        error_metrics = {"Error": "Price or signal data is empty, cannot backtest."}
        yield pd.Series(1.0, index=price_series.index), pd.Series(1.0, index=price_series.index), [], [], error_metrics
        return # Stop the generator

    # --- Main Backtesting Loop ---
    for i in range(len(price_series)):
        current_date = price_series.index[i]
        current_price = price_series.iloc[i]
        signal = signals.iloc[i]

        # --- Update Portfolio Based on Signals ---
        if pd.isna(current_price):
            if i > 0:
                portfolio_value_with_commission.iloc[i] = portfolio_value_with_commission.iloc[i-1]
                portfolio_value_no_commission.iloc[i] = portfolio_value_no_commission.iloc[i-1]
            else:
                portfolio_value_with_commission.iloc[i] = initial_capital
                portfolio_value_no_commission.iloc[i] = initial_capital
        else:
            # Handle Buy Signal
            if signal == 1 and open_trade is None and current_cash_wc > 0:
                shares_to_buy_wc = (current_cash_wc / current_price) * (1 - commission_rate)
                holding_shares_wc = shares_to_buy_wc
                current_cash_wc = 0
                
                holding_shares_nc = current_cash_nc / current_price
                current_cash_nc = 0

                open_trade = {'buy_date': current_date, 'buy_price': current_price, 'shares_wc': holding_shares_wc, 'shares_nc': holding_shares_nc}
                buy_points.append({'date': current_date, 'price': current_price})

            # Handle Sell Signal
            elif signal == -1 and open_trade is not None:
                revenue_wc = open_trade['shares_wc'] * current_price * (1 - commission_rate)
                current_cash_wc = revenue_wc
                
                revenue_nc = open_trade['shares_nc'] * current_price
                current_cash_nc = revenue_nc

                # Log completed trade
                trade_profit_wc = revenue_wc - (open_trade['shares_wc'] / (1 - commission_rate)) * open_trade['buy_price']
                trade_profit_nc = revenue_nc - (open_trade['shares_nc'] * open_trade['buy_price'])
                commission_cost = (open_trade['shares_wc'] / (1 - commission_rate)) * open_trade['buy_price'] * commission_rate + open_trade['shares_wc'] * current_price * commission_rate

                # Calculate profit percentage
                profit_percentage = (trade_profit_wc / (open_trade['shares_wc'] * open_trade['buy_price'])) * 100 if (open_trade['shares_wc'] * open_trade['buy_price']) != 0 else 0

                # Calculate current total capital and return percentage at the time of trade
                current_total_capital_at_trade = revenue_wc # After selling, cash is revenue_wc
                current_total_return_percentage = (current_total_capital_at_trade / initial_capital - 1) * 100


                trades_info.append({
                    '時間': current_date.strftime('%Y-%m-%d %H:%M:%S'), # Use current_date as trade time
                    '幣種': currency.upper(),
                    '買進價格': open_trade['buy_price'],
                    '賣出價格': current_price,
                    '買進時間': open_trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    '賣出時間': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    '持有時間': (current_date - open_trade['buy_date']).days,
                    '利潤': trade_profit_wc,
                    '利潤%數': profit_percentage,
                    '當前總資產': current_total_capital_at_trade,
                    '當前總報酬率%數': current_total_return_percentage,
                    'commission_cost': commission_cost # Re-add commission_cost
                })
                
                holding_shares_wc = 0
                holding_shares_nc = 0
                open_trade = None
                sell_points.append({'date': current_date, 'price': current_price})
            
            # Calculate daily net worth
            if holding_shares_wc > 0:
                portfolio_value_with_commission.iloc[i] = current_cash_wc + (holding_shares_wc * current_price)
                portfolio_value_no_commission.iloc[i] = current_cash_nc + (holding_shares_nc * current_price)
            else:
                portfolio_value_with_commission.iloc[i] = current_cash_wc
                portfolio_value_no_commission.iloc[i] = current_cash_nc

        # --- Calculate Metrics for the Current Snapshot ---
        # Use data up to the current day `i`
        portfolio_snapshot_wc = portfolio_value_with_commission.iloc[:i+1].dropna()
        if portfolio_snapshot_wc.empty:
            continue

        cumulative_returns_wc = portfolio_snapshot_wc / initial_capital
        
        # Calculate Buy and Hold cumulative returns
        buy_and_hold_cumulative_returns = (price_series.iloc[:i+1].dropna() / price_series.iloc[0]) * initial_capital / initial_capital if price_series.iloc[0] > 0 else pd.Series(1.0, index=price_series.iloc[:i+1].dropna().index)
        # Ensure the index matches for plotting
        buy_and_hold_cumulative_returns = buy_and_hold_cumulative_returns.reindex(cumulative_returns_wc.index, fill_value=1.0)

        total_return_wc = (cumulative_returns_wc.iloc[-1] - 1) * 100
        max_peak_wc = cumulative_returns_wc.expanding(min_periods=1).max()
        drawdown_wc = (cumulative_returns_wc / max_peak_wc) - 1
        max_drawdown_wc = drawdown_wc.min() * 100

        # Buy and Hold for the same period
        buy_and_hold_snapshot = price_series.iloc[:i+1].dropna()
        if not buy_and_hold_snapshot.empty and buy_and_hold_snapshot.iloc[0] > 0:
            buy_and_hold_return = (buy_and_hold_snapshot.iloc[-1] / buy_and_hold_snapshot.iloc[0] - 1) * 100
        else:
            buy_and_hold_return = 0.0

        # Trade-specific metrics
        total_trades = len(trades_info)
        winning_trades = [t for t in trades_info if t['利潤'] > 0] # Use '利潤' for profit
        total_profit = sum(t['利潤'] for t in winning_trades)
        losing_trades = [t for t in trades_info if t['利潤'] < 0]
        total_loss = abs(sum(t['利潤'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate new metrics
        average_holding_days = sum(t['持有時間'] for t in trades_info) / total_trades if total_trades > 0 else 0
        average_profit_per_trade = sum(t['利潤'] for t in trades_info) / total_trades if total_trades > 0 else 0
        trade_frequency = total_trades / len(price_series) # Trades per day

        performance_metrics = {
            "Status": f"Processing day {i+1}/{len(price_series)}",
            "Selected Strategy Total Return (with commissions)": f"{total_return_wc:.2f}%",
            "Max Drawdown": f"{max_drawdown_wc:.2f}%",
            "Buy and Hold Strategy Total Return": f"{buy_and_hold_return:.2f}%",
            "Final Capital (with commissions)": f"{portfolio_snapshot_wc.iloc[-1]:.2f} {currency.upper()}",
            "--- Trade Statistics ---": "",
            "Total Trades": total_trades,
            "Win Rate": f"{(len(winning_trades) / total_trades * 100) if total_trades > 0 else 0:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinity",
            "Total Commission Paid": f"{sum(t['commission_cost'] for t in trades_info):.2f} {currency.upper()}",
            "Average Holding Days": f"{average_holding_days:.2f} days",
            "Average Profit per Trade": f"{average_profit_per_trade:.2f} {currency.upper()}",
            "Trade Frequency (per day)": f"{trade_frequency:.4f}"
        }

        # --- Yield the current state ---
        yield cumulative_returns_wc, buy_and_hold_cumulative_returns, buy_points, sell_points, performance_metrics, trades_info