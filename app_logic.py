import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gradio as gr
import io
import sys
import itertools

import config
from data_fetcher import get_crypto_prices, get_github_commits
from strategies import simple_commit_threshold_strategy, commit_sma_strategy, llm_strategy_generator
from strategies.simple_sma_strategy import simple_sma_strategy
from strategies.simple_sma_strategy import simple_sma_strategy
from backtester import run_backtest
from strategies.simple_sma_strategy import simple_sma_strategy

def parse_parameter_input(param_str, param_name):
    """
    Parses a parameter string which can be a single value, a comma-separated list, or a range.
    Examples: "50", "50,60,70", "50-70-5"
    Returns a list of integers.
    """
    if not param_str:
        return []

    param_str = str(param_str).strip()

    if "," in param_str:
        # Comma-separated list
        try:
            return [int(x.strip()) for x in param_str.split(",")]
        except ValueError:
            raise gr.Error(f"Invalid format for {param_name}: '{param_str}'. Use comma-separated numbers (e.g., '50,60,70').")
    elif "-" in param_str and param_str.count("-") == 2:
        # Range format: start-end-step
        try:
            parts = [int(x.strip()) for x in param_str.split("-")]
            if len(parts) != 3:
                raise ValueError("Incorrect number of parts for range.")
            start, end, step = parts
            if step == 0:
                raise ValueError("Step cannot be zero.")
            if (step > 0 and start > end) or (step < 0 and start < end):
                raise ValueError("Step direction is inconsistent with start and end values.")
            return list(range(start, end + (1 if step > 0 else -1), step))
        except ValueError as e:
            raise gr.Error(f"Invalid range format for {param_name}: '{param_str}'. Use start-end-step (e.g., '50-70-5'). Error: {e}")
    else:
        # Single value
        try:
            return [int(param_str)]
        except ValueError:
            raise gr.Error(f"Invalid number format for {param_name}: '{param_str}'. Expected a single number.")

def analyze_crypto_activity(crypto_selection, manual_binance_symbol, manual_owner, manual_repo, 
                            start_date_input, end_date_input, 
                            strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode, 
                            buy_threshold_input, buy_threshold_range_input, 
                            sell_threshold_input, sell_threshold_range_input,
                            short_sma_period_input, short_sma_period_range_input, 
                            long_sma_period_input, long_sma_period_range_input,
                            buy_score_threshold_input, buy_score_threshold_range_input, 
                            sell_score_threshold_input, sell_score_threshold_range_input,
                            sma1_period_input, sma1_period_range_input, 
                            sma2_period_input, sma2_period_range_input,
                            apply_commission_to_plot, enable_dynamic_updates, hyperparameter_search_mode, progress=gr.Progress(track_tqdm=True)):
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

        # Ensure start_dt and end_dt are datetime objects, not Gradio DateTime components
        if isinstance(start_dt, gr.components.DateTime):
            start_dt = start_dt.value
        if isinstance(end_dt, gr.components.DateTime):
            end_dt = end_dt.value

        # If they are still strings, try to parse them into datetime objects
        if isinstance(start_dt, str):
            try:
                start_dt = datetime.fromisoformat(start_dt)
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Could not parse start_dt string: {start_dt}. Falling back to default.")
                    start_dt = datetime.now() - timedelta(days=config.DEFAULT_DAYS)

        if isinstance(end_dt, str):
            try:
                end_dt = datetime.fromisoformat(end_dt)
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Could not parse end_dt string: {end_dt}. Falling back to default.")
                    end_dt = datetime.now()

        print(f"DEBUG: Final start_dt type: {type(start_dt)}, value: {start_dt}")
        print(f"DEBUG: Final end_dt type: {type(end_dt)}, value: {end_dt}")
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
                yield None, None, "Error: Could not fetch necessary data.", "Analysis failed.", redirected_output_capture.getvalue(), [], []
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

        def generate_ui_output(results, is_hyperparameter_search_mode=False, strategy_choice=None, price_series_aligned=None, commit_counts_for_plot=None):
            if is_hyperparameter_search_mode:
                # In hyperparameter search mode, results is the sorted_results list
                # We need to extract the table data from it
                results_table_data = []
                for res in results:
                    params = res["params"]
                    metrics = res["performance_metrics"]
                    print(f"DEBUG: Raw metrics for combo {params.get('combo_index', '')}: {metrics}")
                    
                    # Extract and convert values, handling potential errors
                    total_return_str = metrics.get("Selected Strategy Total Return (with commissions)", "0%")
                    annualized_return_str = metrics.get("Annualized Return (%)", "0%")
                    max_drawdown_str = metrics.get("Max Drawdown", "0%")
                    sharpe_ratio_str = metrics.get("Sharpe Ratio", "0")
                    sortino_ratio_str = metrics.get("Sortino Ratio", "0")
                    win_rate_str = metrics.get("Win Rate", "0%")
                    total_trades = metrics.get("Total Trades", 0)

                    try:
                        total_return = float(total_return_str.replace('%', ''))
                    except ValueError:
                        total_return = 0.0
                    
                    try:
                        annualized_return = float(annualized_return_str.replace('%', ''))
                    except ValueError:
                        annualized_return = 0.0

                    try:
                        max_drawdown = float(max_drawdown_str.replace('%', ''))
                    except ValueError:
                        max_drawdown = 0.0

                    try:
                        sharpe_ratio = float(sharpe_ratio_str)
                    except ValueError:
                        sharpe_ratio = 0.0

                    try:
                        sortino_ratio = float(sortino_ratio_str)
                    except ValueError:
                        sortino_ratio = 0.0

                    try:
                        win_rate = float(win_rate_str.replace('%', ''))
                    except ValueError:
                        win_rate = 0.0

                    print(f"DEBUG: Extracted and converted values - Total Return: {total_return}, Annualized Return: {annualized_return}, Max Drawdown: {max_drawdown}, Sharpe: {sharpe_ratio}, Sortino: {sortino_ratio}, Win Rate: {win_rate}, Total Trades: {total_trades}")

                    row = [
                        params.get("combo_index", ""),
                        total_return,
                        annualized_return,
                        max_drawdown,
                        sharpe_ratio,
                        sortino_ratio,
                        win_rate,
                        total_trades,
                        f"BuyThresh: {params['buy_threshold']}, SellThresh: {params['sell_threshold']}" if "Simple Commit Threshold Strategy" in strategy_choice else "",
                        f"ShortSMA: {params['short_sma']}, LongSMA: {params['long_sma']}" if "Commit SMA Strategy" in strategy_choice else "",
                        f"BuyScore: {params['buy_score']}, SellScore: {params['sell_score']}" if "LLM Commit Analysis Strategy" in strategy_choice else "",
                        f"SMA1: {params['sma1']}, SMA2: {params['sma2']}" if "Simple SMA Strategy" in strategy_choice else ""
                    ]
                    print(f"DEBUG: Row data being appended for combo {params.get('combo_index', '')}: {row}")
                    results_table_data.append(row)
                
                return None, None, "Hyperparameter Search Results", "Hyperparameter search completed. See table below.", redirected_output_capture.getvalue(), [], results_table_data
            else:
                # Original single backtest mode output
                cumulative_returns_wc = results["cumulative_returns_wc"]
                buy_and_hold_returns_plot = results["buy_and_hold_returns_plot"]
                buy_points = results["buy_points"]
                sell_points = results["sell_points"]
                performance_metrics = results["performance_metrics"]
                trades_info_raw = results["trades_info_raw"]

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
                return fig1, fig2, performance_text, performance_metrics.get("Status", "Running..."), redirected_output_capture.getvalue(), trades_info_formatted, None

        # Determine which parameters to use based on hyperparameter_search_mode
        param_configs = []
        if hyperparameter_search_mode:
            print("DEBUG: Hyperparameter search mode enabled. Parsing range inputs.")
            # Parse all range inputs
            buy_threshold_values = parse_parameter_input(buy_threshold_range_input, "Buy Commit Threshold") if "Simple Commit Threshold Strategy" in strategy_choice else [buy_threshold_input]
            sell_threshold_values = parse_parameter_input(sell_threshold_range_input, "Sell Commit Threshold") if "Simple Commit Threshold Strategy" in strategy_choice else [sell_threshold_input]
            short_sma_values = parse_parameter_input(short_sma_period_range_input, "Short Commit SMA Period") if "Commit SMA Strategy" in strategy_choice else [short_sma_period_input]
            long_sma_values = parse_parameter_input(long_sma_period_range_input, "Long Commit SMA Period") if "Commit SMA Strategy" in strategy_choice else [long_sma_period_input]
            buy_score_values = parse_parameter_input(buy_score_threshold_range_input, "Buy Score Threshold") if "LLM Commit Analysis Strategy" in strategy_choice else [buy_score_threshold_input]
            sell_score_values = parse_parameter_input(sell_score_threshold_range_input, "Sell Score Threshold") if "LLM Commit Analysis Strategy" in strategy_choice else [sell_score_threshold_input]
            sma1_values = parse_parameter_input(sma1_period_range_input, "SMA1 Period") if "Simple SMA Strategy" in strategy_choice else [sma1_period_input]
            sma2_values = parse_parameter_input(sma2_period_range_input, "SMA2 Period") if "Simple SMA Strategy" in strategy_choice else [sma2_period_input]

            print(f"DEBUG: buy_threshold_values: {buy_threshold_values}")
            print(f"DEBUG: sell_threshold_values: {sell_threshold_values}")
            print(f"DEBUG: short_sma_values: {short_sma_values}")
            print(f"DEBUG: long_sma_values: {long_sma_values}")
            print(f"DEBUG: buy_score_values: {buy_score_values}")
            print(f"DEBUG: sell_score_values: {sell_score_values}")
            print(f"DEBUG: sma1_values: {sma1_values}")
            print(f"DEBUG: sma2_values: {sma2_values}")

            # Generate all combinations
            param_combinations = list(itertools.product(
                buy_threshold_values, sell_threshold_values,
                short_sma_values, long_sma_values,
                buy_score_values, sell_score_values,
                sma1_values, sma2_values
            ))

            for i, combo in enumerate(param_combinations):
                param_configs.append({
                    "buy_threshold": combo[0],
                    "sell_threshold": combo[1],
                    "short_sma": combo[2],
                    "long_sma": combo[3],
                    "buy_score": combo[4],
                    "sell_score": combo[5],
                    "sma1": combo[6],
                    "sma2": combo[7],
                    "combo_index": i + 1 # For display purposes
                })
            print(f"DEBUG: Generated {len(param_configs)} hyperparameter combinations.")
            if not param_configs:
                raise gr.Error("No valid parameter combinations generated. Please check your input ranges.")

        else:
            # Single run mode
            print("DEBUG: Single run mode enabled.")
            param_configs.append({
                "buy_threshold": buy_threshold_input,
                "sell_threshold": sell_threshold_input,
                "short_sma": short_sma_period_input,
                "long_sma": long_sma_period_input,
                "buy_score": buy_score_threshold_input,
                "sell_score": sell_score_threshold_input,
                "sma1": sma1_period_input,
                "sma2": sma2_period_input,
                "combo_index": 1
            })

        all_backtest_results = []
        total_combinations = len(param_configs)

        for combo_idx, params in enumerate(param_configs):
            current_progress = (combo_idx / total_combinations) * 100
            progress(current_progress / 100, desc=f"Running backtest for combination {combo_idx + 1}/{total_combinations} with params: {params}...")
            print(f"DEBUG: Running backtest for combination {combo_idx + 1}/{total_combinations} with params: {params}")

            # Generate signals for each selected strategy with current parameters
            all_strategy_signals = pd.DataFrame(index=price_series_aligned.index)
            all_strategy_holdings = pd.DataFrame(index=price_series_aligned.index, dtype=bool)

            if "No Strategy" in strategy_choice or not strategy_choice:
                final_strategy_signals = pd.Series(0, index=price_series_aligned.index, dtype=int)
            else:
                for strategy in strategy_choice:
                    signals = pd.Series(0, index=price_series_aligned.index, dtype=int) # Initialize signals for current strategy
                    if strategy == "Simple Commit Threshold Strategy":
                        signals = simple_commit_threshold_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), params["buy_threshold"], params["sell_threshold"])
                    elif strategy == "Commit SMA Strategy":
                        signals = commit_sma_strategy(commit_counts_for_plot.reindex(price_series_aligned.index, fill_value=0), params["short_sma"], params["long_sma"])
                    elif strategy == "LLM Commit Analysis Strategy":
                        llm_signals_generator = llm_strategy_generator(commit_df, price_series_aligned, 
                                                                    params["buy_score"], params["sell_score"],
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
                        signals = simple_sma_strategy(price_series_aligned, params["sma1"], params["sma2"])

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

            # Run backtest to completion for each combination
            final_results = list(backtest_generator)[-1]
            cumulative_returns_wc, buy_and_hold_returns_plot, buy_points, sell_points, performance_metrics, trades_info_raw = final_results
            
            # Store results for this combination
            all_backtest_results.append({
                "params": params,
                "performance_metrics": performance_metrics,
                "cumulative_returns_wc": cumulative_returns_wc,
                "buy_and_hold_returns_plot": buy_and_hold_returns_plot,
                "buy_points": buy_points,
                "sell_points": sell_points,
                "trades_info_raw": trades_info_raw
            })
            print(f"DEBUG: Backtest for combination {combo_idx + 1} completed. Performance metrics: {performance_metrics}")
        
        # After all backtests are run, process and display results
        if hyperparameter_search_mode:
            print("DEBUG: All backtests completed in hyperparameter search mode. Sorting results.")
            # Sort results by final return rate
            sorted_results = sorted(all_backtest_results, key=lambda x: x["performance_metrics"].get("Selected Strategy Total Return (with commissions)", "-inf").replace('%', '') if isinstance(x["performance_metrics"].get("Selected Strategy Total Return (with commissions)"), str) else -float("inf"), reverse=True)

            # Prepare data for the results table
            results_table_data = []
            for res in sorted_results:
                params = res["params"]
                metrics = res["performance_metrics"]
                print(f"DEBUG: Raw metrics for combo {params.get('combo_index', '')}: {metrics}")
                
                # Extract and convert values, handling potential errors
                total_return_str = metrics.get("Selected Strategy Total Return (with commissions)", "0%")
                annualized_return_str = metrics.get("Annualized Return (%)", "0%")
                max_drawdown_str = metrics.get("Max Drawdown", "0%")
                sharpe_ratio_str = metrics.get("Sharpe Ratio", "0")
                sortino_ratio_str = metrics.get("Sortino Ratio", "0")
                win_rate_str = metrics.get("Win Rate", "0%")
                total_trades = metrics.get("Total Trades", 0)

                try:
                    total_return = float(total_return_str.replace('%', ''))
                except ValueError:
                    total_return = 0.0
                
                try:
                    annualized_return = float(annualized_return_str.replace('%', ''))
                except ValueError:
                    annualized_return = 0.0

                try:
                    max_drawdown = float(max_drawdown_str.replace('%', ''))
                except ValueError:
                    max_drawdown = 0.0

                try:
                    sharpe_ratio = float(sharpe_ratio_str)
                except ValueError:
                    sharpe_ratio = 0.0

                try:
                    sortino_ratio = float(sortino_ratio_str)
                except ValueError:
                    sortino_ratio = 0.0

                try:
                    win_rate = float(win_rate_str.replace('%', ''))
                except ValueError:
                    win_rate = 0.0

                print(f"DEBUG: Extracted and converted values - Total Return: {total_return}, Annualized Return: {annualized_return}, Max Drawdown: {max_drawdown}, Sharpe: {sharpe_ratio}, Sortino: {sortino_ratio}, Win Rate: {win_rate}, Total Trades: {total_trades}")

                row = [
                    params.get("combo_index", ""),
                    total_return,
                    annualized_return,
                    max_drawdown,
                    sharpe_ratio,
                    sortino_ratio,
                    win_rate,
                    total_trades,
                    f"BuyThresh: {params['buy_threshold']}, SellThresh: {params['sell_threshold']}" if "Simple Commit Threshold Strategy" in strategy_choice else "",
                    f"ShortSMA: {params['short_sma']}, LongSMA: {params['long_sma']}" if "Commit SMA Strategy" in strategy_choice else "",
                    f"BuyScore: {params['buy_score']}, SellScore: {params['sell_score']}" if "LLM Commit Analysis Strategy" in strategy_choice else "",
                    f"SMA1: {params['sma1']}, SMA2: {params['sma2']}" if "Simple SMA Strategy" in strategy_choice else ""
                ]
                print(f"DEBUG: Row data being appended for combo {params.get('combo_index', '')}: {row}")
                results_table_data.append(row)
            
            # For hyperparameter search mode, we yield a table and a message
            yield generate_ui_output(sorted_results, is_hyperparameter_search_mode=True, strategy_choice=strategy_choice)
            
        else:
            # Original single backtest mode output
            print("DEBUG: Dynamic updates disabled. Running full backtest...")
            progress(0.5, desc="Running backtest...")
            final_results = all_backtest_results[0] # Get the single result
            print("DEBUG: Full backtest completed. Generating final charts...")
            progress(1.0, desc="Generating final charts...")
            yield generate_ui_output(final_results, is_hyperparameter_search_mode=False, strategy_choice=strategy_choice, price_series_aligned=price_series_aligned, commit_counts_for_plot=commit_counts_for_plot)
            print("DEBUG: Yielded final UI output.")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr