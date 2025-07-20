import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gradio as gr
import io
import sys

import config
from data_fetcher import get_crypto_prices, get_github_commits
from strategies import simple_commit_threshold_strategy, commit_sma_strategy, llm_strategy_generator
from backtester import run_backtest

def analyze_crypto_activity(crypto_selection, manual_coingecko_id, manual_owner, manual_repo, 
                            start_date_input, end_date_input, 
                            strategy_choice, buy_logic, sell_logic, buy_threshold_input, sell_threshold_input,
                            short_sma_period_input, long_sma_period_input,
                            buy_score_threshold_input, sell_score_threshold_input,
                            apply_commission_to_plot, enable_dynamic_updates, progress=gr.Progress(track_tqdm=True)):
    print("DEBUG: analyze_crypto_activity function entered.")
    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output

    try:
        progress(0, desc="Initializing and fetching data...")
        start_dt = start_date_input if start_date_input is not None else datetime.now() - timedelta(days=config.DEFAULT_DAYS)
        end_dt = end_date_input if end_date_input is not None else datetime.now()

        coingecko_id_to_use, github_owner_to_use, github_repo_to_use = "", "", ""
        if crypto_selection == "Manual GitHub Repo & CoinGecko ID":
            coingecko_id_to_use, github_owner_to_use, github_repo_to_use = manual_coingecko_id, manual_owner, manual_repo
        elif crypto_selection in config.PREDEFINED_CRYPTOS:
            crypto_info = config.PREDEFINED_CRYPTOS[crypto_selection]
            coingecko_id_to_use, github_owner_to_use, github_repo_to_use = crypto_info["coingecko_id"], crypto_info["github_owner"], crypto_info["github_repo"]
        else:
            coingecko_id_to_use, github_owner_to_use, github_repo_to_use = crypto_selection, manual_owner, manual_repo

        price_series = get_crypto_prices(coingecko_id_to_use, config.DEFAULT_CRYPTO_CURRENCY, start_dt, end_dt)
        commit_df = get_github_commits(github_owner_to_use, github_repo_to_use, start_dt, end_dt, config.headers)

        if price_series.empty or commit_df.empty:
                yield None, None, "Error: Could not fetch necessary data.", "Analysis failed.", redirected_output.getvalue()
                return

        floored_start_date = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        floored_end_date = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        full_date_range = pd.date_range(start=floored_start_date, end=floored_end_date, freq='D')
        commit_counts_for_plot = commit_df.groupby(commit_df['date'].dt.date).size().reindex(full_date_range.date, fill_value=0)
        price_series_aligned = price_series.reindex(full_date_range).dropna()

        # New function to combine strategy signals
        def combine_strategy_signals(all_signals, buy_logic, sell_logic):
            combined_buy_signals = pd.Series(False, index=all_signals.index)
            combined_sell_signals = pd.Series(False, index=all_signals.index)

            if not all_signals.empty:
                # Combine buy signals
                if buy_logic == "AND":
                    combined_buy_signals = all_signals.apply(lambda row: all(row[col] == 1 for col in all_signals.columns), axis=1)
                else: # OR logic
                    combined_buy_signals = all_signals.apply(lambda row: any(row[col] == 1 for col in all_signals.columns), axis=1)

                # Combine sell signals
                if sell_logic == "AND":
                    combined_sell_signals = all_signals.apply(lambda row: all(row[col] == -1 for col in all_signals.columns), axis=1)
                else: # OR logic
                    combined_sell_signals = all_signals.apply(lambda row: any(row[col] == -1 for col in all_signals.columns), axis=1)

            final_signals = pd.Series(0, index=all_signals.index, dtype=int)
            final_signals[combined_buy_signals] = 1
            final_signals[combined_sell_signals] = -1
            
            # Ensure buy and sell signals don't overlap for the same day
            # If both buy and sell signals are true for a day, prioritize sell (or buy, depending on desired behavior) 
            # Here, we'll make it 0 (no action) if both are true
            final_signals[(combined_buy_signals) & (combined_sell_signals)] = 0

            return final_signals

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
            return fig1, fig2, performance_text, performance_metrics.get("Status", "Running..."), redirected_output.getvalue(), trades_info_formatted

        # Generate signals for each selected strategy
        all_strategy_signals = pd.DataFrame(index=price_series_aligned.index)

        if "No Strategy" in strategy_choice or not strategy_choice:
            # If "No Strategy" is selected or no strategy is selected, return no signals
            final_strategy_signals = pd.Series(0, index=price_series_aligned.index, dtype=int)
        else:
            for strategy in strategy_choice:
                if strategy == "Simple Commit Threshold Strategy":
                    signals = simple_commit_threshold_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), buy_threshold_input, sell_threshold_input)
                    all_strategy_signals["Simple Commit Threshold Strategy"] = signals
                elif strategy == "Commit SMA Strategy":
                    signals = commit_sma_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), short_sma_period_input, long_sma_period_input)
                    all_strategy_signals["Commit SMA Strategy"] = signals
                elif strategy == "LLM Commit Analysis Strategy":
                    # LLM strategy will generate signals for the entire period first
                    # The llm_strategy_generator will return a single series of signals
                    llm_signals_generator = llm_strategy_generator(commit_df, price_series_aligned, 
                                                                buy_score_threshold_input, sell_score_threshold_input,
                                                                config.INITIAL_CAPITAL, config.COMMISSION_RATE, 
                                                                config.DEFAULT_CRYPTO_CURRENCY, progress, return_signals_only=True)
                    # Consume the generator to get the final signals
                    llm_signals = None
                    for _, _, _, _, _, signals_only in llm_signals_generator:
                        llm_signals = signals_only # The last yield will contain the full signals
                    
                    if llm_signals is not None:
                        all_strategy_signals["LLM Commit Analysis Strategy"] = llm_signals.reindex(price_series_aligned.index, fill_value=0)
                    else:
                        all_strategy_signals["LLM Commit Analysis Strategy"] = pd.Series(0, index=price_series_aligned.index, dtype=int)

            # Combine signals based on selected logic
            final_strategy_signals = combine_strategy_signals(all_strategy_signals, buy_logic, sell_logic)

            backtest_generator = run_backtest(price_series_aligned, final_strategy_signals, config.INITIAL_CAPITAL, config.COMMISSION_RATE, config.DEFAULT_CRYPTO_CURRENCY)

            if enable_dynamic_updates:
                for i, results in enumerate(backtest_generator):
                    progress(0.5 + (i / len(price_series_aligned) * 0.5), desc=results[4].get("Status", "Backtesting..."))
                    yield generate_ui_output(results)
            else:
                progress(0.5, desc="Running backtest...")
                final_results = list(backtest_generator)[-1]
                progress(1.0, desc="Generating final charts...")
                yield generate_ui_output(final_results)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr