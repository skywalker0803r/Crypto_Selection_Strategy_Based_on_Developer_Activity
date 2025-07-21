import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gradio as gr
import io
import sys

import config
from data_fetcher import get_crypto_prices, get_github_commits
from strategies import simple_commit_threshold_strategy, commit_sma_strategy, llm_strategy_generator
from strategies.simple_sma_strategy import simple_sma_strategy
from strategies.simple_sma_strategy import simple_sma_strategy
from backtester import run_backtest
from strategies.simple_sma_strategy import simple_sma_strategy

def analyze_crypto_activity(crypto_selection, manual_binance_symbol, manual_owner, manual_repo, 
                            start_date_input, end_date_input, 
                            strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode, buy_threshold_input, sell_threshold_input,
                            short_sma_period_input, long_sma_period_input,
                            buy_score_threshold_input, sell_score_threshold_input,
                            sma1_period_input, sma2_period_input,
                            apply_commission_to_plot, enable_dynamic_updates, progress=gr.Progress(track_tqdm=True)):
    print("DEBUG: analyze_crypto_activity function entered.")
    # Capture stdout/stderr
    # Custom class to tee output to both StringIO and original stdout/stderr
    class DualOutput:
        def __init__(self, file1, file2):
            self.file1 = file1
            self.file2 = file2

        def write(self, s):
            self.file1.write(s)
            self.file2.write(s)

        def flush(self):
            self.file1.flush()
            self.file2.flush()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output_capture = io.StringIO()
    sys.stdout = DualOutput(redirected_output_capture, old_stdout)
    sys.stderr = DualOutput(redirected_output_capture, old_stderr)

    try:
        progress(0, desc="Initializing and fetching data...")
        print(f"DEBUG: Received start_date_input: {start_date_input}, type: {type(start_date_input)}")
        print(f"DEBUG: Received end_date_input: {end_date_input}, type: {type(end_date_input)}")

        start_dt = start_date_input if start_date_input is not None else datetime.now() - timedelta(days=config.DEFAULT_DAYS)
        end_dt = end_date_input if end_date_input is not None else datetime.now()
        print(f"DEBUG: Using start_dt: {start_dt}, end_dt: {end_dt}")

        binance_symbol_to_use, github_owner_to_use, github_repo_to_use = "", "", ""
        if crypto_selection == "Manual GitHub Repo & Binance Symbol":
            binance_symbol_to_use, github_owner_to_use, github_repo_to_use = manual_binance_symbol, manual_owner, manual_repo
        elif crypto_selection in config.PREDEFINED_CRYPTOS:
            crypto_info = config.PREDEFINED_CRYPTOS[crypto_selection]
            binance_symbol_to_use, github_owner_to_use, github_repo_to_use = crypto_info["binance_symbol"], crypto_info["github_owner"], crypto_info["github_repo"]
        else:
            binance_symbol_to_use, github_owner_to_use, github_repo_to_use = crypto_selection, manual_owner, manual_repo

        price_series = get_crypto_prices(binance_symbol_to_use, config.DEFAULT_CRYPTO_CURRENCY, start_dt, end_dt)
        print(f"DEBUG: Fetched {len(price_series)} days of price data.")
        commit_df = get_github_commits(github_owner_to_use, github_repo_to_use, start_dt, end_dt, config.headers)
        print(f"DEBUG: Fetched {len(commit_df)} GitHub commit data points.")

        if price_series.empty or commit_df.empty:
                yield None, None, "Error: Could not fetch necessary data.", "Analysis failed.", redirected_output_capture.getvalue()
                return

        floored_start_date = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        floored_end_date = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        full_date_range = pd.date_range(start=floored_start_date, end=floored_end_date, freq='D')
        print(f"DEBUG: Full date range for analysis: {floored_start_date.strftime('%Y-%m-%d')} to {floored_end_date.strftime('%Y-%m-%d')} ({len(full_date_range)} days).")
        commit_counts_for_plot = commit_df.groupby(commit_df['date'].dt.date).size().reindex(full_date_range.date, fill_value=0)
        print(f"DEBUG: Daily commit counts prepared for plotting. Total days with commits: {len(commit_counts_for_plot[commit_counts_for_plot > 0])}.")
        price_series_aligned = price_series.reindex(full_date_range).dropna()
        print(f"DEBUG: Price series aligned to full date range. Remaining data points: {len(price_series_aligned)}.")

        # New function to combine strategy signals
        def combine_strategy_signals(all_signals, buy_logic, sell_logic, all_holdings, buy_combination_mode, sell_combination_mode):
            combined_buy_signals = pd.Series(False, index=all_signals.index)
            combined_sell_signals = pd.Series(False, index=all_signals.index)

            if not all_signals.empty:
                # Combine buy signals
                if buy_logic == "AND":
                    if buy_combination_mode == "同時": # Original AND logic
                        combined_buy_signals = all_signals.apply(lambda row: all(row[col] == 1 for col in all_signals.columns), axis=1)
                    else: # "非同時" (Non-simultaneous) logic
                        for idx, row in all_signals.iterrows():
                            current_date = idx
                            buy_conditions_met = False
                            
                            strategies_with_buy_signal = [col for col in all_signals.columns if row[col] == 1]
                            
                            if strategies_with_buy_signal:
                                for primary_strategy in strategies_with_buy_signal:
                                    all_others_meet_condition = True
                                    for other_strategy in all_signals.columns:
                                        if other_strategy == primary_strategy:
                                            continue
                                        
                                        if not (row[other_strategy] == 1 or all_holdings[other_strategy].loc[current_date] == True):
                                            all_others_meet_condition = False
                                            break
                                if all_others_meet_condition:
                                    buy_conditions_met = True
                                    break
                            
                            combined_buy_signals.loc[current_date] = buy_conditions_met

                else: # OR logic (remains the same)
                    combined_buy_signals = all_signals.apply(lambda row: any(row[col] == 1 for col in all_signals.columns), axis=1)

                # Combine sell signals
                if sell_logic == "AND":
                    if sell_combination_mode == "同時": # Original AND logic
                        combined_sell_signals = all_signals.apply(lambda row: all(row[col] == -1 for col in all_signals.columns), axis=1)
                    else: # "非同時" (Non-simultaneous) logic
                        for idx, row in all_signals.iterrows():
                            current_date = idx
                            sell_conditions_met = False
                            
                            strategies_with_sell_signal = [col for col in all_signals.columns if row[col] == -1]
                            
                            if strategies_with_sell_signal:
                                for primary_strategy in strategies_with_sell_signal:
                                    all_others_meet_condition = True
                                    for other_strategy in all_signals.columns:
                                        if other_strategy == primary_strategy:
                                            continue
                                        
                                        # Condition for other strategies: either they also sell OR they are NOT holding (i.e., in cash/flat)
                                        if not (row[other_strategy] == -1 or all_holdings[other_strategy].loc[current_date] == False):
                                            all_others_meet_condition = False
                                            break
                                    if all_others_meet_condition:
                                        sell_conditions_met = True
                                        break
                            
                            combined_sell_signals.loc[current_date] = sell_conditions_met

                else: # OR logic (remains the same)
                    combined_sell_signals = all_signals.apply(lambda row: any(row[col] == -1 for col in all_signals.columns), axis=1)

            final_signals = pd.Series(0, index=all_signals.index, dtype=int)
            final_signals[combined_buy_signals] = 1
            final_signals[combined_sell_signals] = -1
            
            final_signals[(combined_buy_signals) & (combined_sell_signals)] = 0

            return final_signals

        def simulate_holdings(price_series, signals_series, initial_capital, commission_rate):
            """
            Simulates trades for a single strategy to determine holding status on each day.
            Returns a Series of booleans (True if holding, False otherwise).
            """
            is_holding_series = pd.Series(False, index=price_series.index, dtype=bool)
            current_cash = initial_capital
            holding_shares = 0
            open_trade = None

            for i, (current_date, current_price) in enumerate(price_series.items()):
                signal = signals_series.loc[current_date]

                if pd.isna(current_price):
                    is_holding_series.loc[current_date] = (holding_shares > 0)
                    continue

                if signal == 1 and open_trade is None and current_cash > 0:
                    shares_to_buy = (current_cash / current_price) * (1 - commission_rate)
                    holding_shares = shares_to_buy
                    current_cash = 0
                    open_trade = {'buy_date': current_date, 'buy_price': current_price, 'shares': holding_shares}
                elif signal == -1 and open_trade is not None:
                    current_cash = holding_shares * current_price * (1 - commission_rate)
                    holding_shares = 0
                    open_trade = None

                is_holding_series.loc[current_date] = (holding_shares > 0)

            return is_holding_series

        def generate_ui_output(results):
            cumulative_returns_wc, buy_and_hold_returns_plot, buy_points, sell_points, performance_metrics, trades_info_raw = results

            # Convert trades_info_raw (list of dicts) to list of lists for gr.Dataframe
            # Ensure the order matches the headers defined in GitHub開發活動追蹤器.py
            # headers=["時間", "幣種", "買進價格", "賣出價格", "買進時間", "賣出時間", "持有時間", "利潤", "利潤%數", "當前總資產", "當前總報酬率%數"]
            trades_info_formatted = []
            for trade in trades_info_raw:
                trades_info_formatted.append([
                    trade.get('時間', ''),
                    trade.get('幣種', ''),
                    f"{trade.get('買進價格', 0):.2f}",
                    f"{trade.get('賣出價格', 0):.2f}",
                    trade.get('買進時間', ''),
                    trade.get('賣出時間', ''),
                    trade.get('持有時間', 0),
                    f"{trade.get('利潤', 0):.2f}",
                    f"{trade.get('利潤%數', 0):.2f}%",
                    f"{trade.get('當前總資產', 0):.2f}",
                    f"{trade.get('當前總報酬率%數', 0):.2f}%"
                ])

            plt.close('all') # Close all existing figures to prevent memory issues
            fig1, ax1 = plt.subplots(figsize=(14, 8))
            ax1.plot(price_series_aligned.index, price_series_aligned.values, color='tab:blue', label='Price Trend')
            ax1.scatter([p['date'] for p in buy_points], [p['price'] for p in buy_points], marker='^', s=100, color='green', label='Buy Point', zorder=5)
            ax1.scatter([p['date'] for p in sell_points], [p['price'] for p in sell_points], marker='v', s=100, color='red', label='Sell Point', zorder=5)
            ax2 = ax1.twinx()
            ax2.bar(commit_counts_for_plot.index, commit_counts_for_plot.values, color='tab:red', alpha=0.6, label='Daily Commits')
            fig1.tight_layout()

            fig2, ax_ret = plt.subplots(figsize=(14, 6))
            ax_ret.plot(cumulative_returns_wc.index, cumulative_returns_wc.values, color='purple', label='Strategy Return')
            ax_ret.plot(buy_and_hold_returns_plot.index, buy_and_hold_returns_plot.values, color='orange', linestyle='--', label='Buy and Hold Return')
            ax_ret.legend()
            fig2.tight_layout()

            performance_text = "\n".join([f"{k}: {v}" for k, v in performance_metrics.items()])        
            return fig1, fig2, performance_text, performance_metrics.get("Status", "Running..."), redirected_output_capture.getvalue(), trades_info_formatted

        # Generate signals for each selected strategy
        all_strategy_signals = pd.DataFrame(index=price_series_aligned.index)
        all_strategy_holdings = pd.DataFrame(index=price_series_aligned.index, dtype=bool)

        if "No Strategy" in strategy_choice or not strategy_choice:
            # If "No Strategy" is selected or no strategy is selected, return no signals
            final_strategy_signals = pd.Series(0, index=price_series_aligned.index, dtype=int)
        else:
            for strategy in strategy_choice:
                signals = pd.Series(0, index=price_series_aligned.index, dtype=int) # Initialize signals for current strategy
                if strategy == "Simple Commit Threshold Strategy":
                    signals = simple_commit_threshold_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), buy_threshold_input, sell_threshold_input)
                elif strategy == "Commit SMA Strategy":
                    signals = commit_sma_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), short_sma_period_input, long_sma_period_input)
                elif strategy == "LLM Commit Analysis Strategy":
                    llm_signals_generator = llm_strategy_generator(commit_df, price_series_aligned, 
                                                                buy_score_threshold_input, sell_score_threshold_input,
                                                                config.INITIAL_CAPITAL, config.COMMISSION_RATE, 
                                                                config.DEFAULT_CRYPTO_CURRENCY, progress, return_signals_only=True)
                    llm_signals = None
                    for _, _, _, _, _, signals_only in llm_signals_generator:
                        llm_signals = signals_only # The last yield will contain the full signals
                    print(f"DEBUG: LLM signals consumed. Length: {len(llm_signals) if llm_signals is not None else 0}")
                    
                    if llm_signals is not None:
                        signals = llm_signals.reindex(price_series_aligned.index, fill_value=0)
                    else:
                        signals = pd.Series(0, index=price_series_aligned.index, dtype=int)
                elif strategy == "Simple SMA Strategy":
                    signals = simple_sma_strategy(price_series_aligned, sma1_period_input, sma2_period_input)

                all_strategy_signals[strategy] = signals
                # Simulate holdings for the current strategy
                holding_status = simulate_holdings(price_series_aligned, signals, config.INITIAL_CAPITAL, config.COMMISSION_RATE)
                all_strategy_holdings[strategy] = holding_status

            # Combine signals based on selected logic
            print("DEBUG: Combining strategy signals...")
            final_strategy_signals = combine_strategy_signals(all_strategy_signals, buy_logic, sell_logic, all_strategy_holdings, buy_combination_mode, sell_combination_mode)
            print(f"DEBUG: Combined strategy signals. Length: {len(final_strategy_signals)}")

            print("DEBUG: Running backtest...")
            backtest_generator = run_backtest(price_series_aligned, final_strategy_signals, config.INITIAL_CAPITAL, config.COMMISSION_RATE, config.DEFAULT_CRYPTO_CURRENCY)
            print("DEBUG: Backtest generator created.")

            if enable_dynamic_updates:
                print("DEBUG: Dynamic updates enabled. Iterating through backtest generator...")
                for i, results in enumerate(backtest_generator):
                    progress(0.5 + (i / len(price_series_aligned) * 0.5), desc=results[4].get("Status", "Backtesting..."))
                    print(f"DEBUG: Yielding UI output for dynamic update (day {i+1})...")
                    yield generate_ui_output(results)
                print("DEBUG: Finished iterating through backtest generator for dynamic updates.")
            else:
                print("DEBUG: Dynamic updates disabled. Running full backtest...")
                progress(0.5, desc="Running backtest...")
                final_results = list(backtest_generator)[-1]
                print("DEBUG: Full backtest completed. Generating final charts...")
                progress(1.0, desc="Generating final charts...")
                yield generate_ui_output(final_results)
                print("DEBUG: Yielded final UI output.")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr