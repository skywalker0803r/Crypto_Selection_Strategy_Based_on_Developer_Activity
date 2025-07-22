import gradio as gr
from datetime import datetime, timedelta

import config
from data_fetcher import get_binance_trading_pairs
from app_logic import analyze_crypto_activity
from dotenv import load_dotenv

load_dotenv()

# --- UI Initialization ---

# Fetch the dynamic list once when the script starts
DYNAMIC_BINANCE_SYMBOLS = get_binance_trading_pairs(100, config.DEFAULT_CRYPTO_CURRENCY, config.PREDEFINED_CRYPTOS)

# Construct the final list for the dropdown: PREDEFINED + DYNAMIC + Manual Option
ALL_DROPDOWN_CHOICES = list(config.PREDEFINED_CRYPTOS.keys()) + DYNAMIC_BINANCE_SYMBOLS + ["Manual GitHub Repo & Binance Symbol"]
# Fallback if ALL_DROPDOWN_CHOICES becomes empty unexpectedly
if not ALL_DROPDOWN_CHOICES:
    ALL_DROPDOWN_CHOICES = ["bitcoin", "ethereum", "solana", "Manual GitHub Repo & Binance Symbol"]
    print("Warning: Dropdown choices are empty, using default fallback list.")


# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Cryptocurrency Price & GitHub Development Activity Tracker with Backtesting
        Explore the relationship between cryptocurrency price trends and daily GitHub development commits by selecting a cryptocurrency or manually entering a GitHub repository.
        Includes strategy backtesting functionality, showing buy/sell points and strategy returns.
        """
    )

    with gr.Row():
        with gr.Column():
            crypto_choice = gr.Dropdown(
                label="Select Cryptocurrency (predefined first, then top 100 by market cap) or Manually Enter GitHub Repo & CoinGecko ID",
                choices=ALL_DROPDOWN_CHOICES,
                value=list(config.PREDEFINED_CRYPTOS.keys())[0] if config.PREDEFINED_CRYPTOS else (ALL_DROPDOWN_CHOICES[0] if ALL_DROPDOWN_CHOICES else None),
                interactive=True
            )
            manual_coingecko_id = gr.Textbox(
                label="Manual Binance Symbol (e.g., DOGEUSDT)",
                placeholder="Enter Binance Symbol",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_owner = gr.Textbox(
                label="GitHub Project Owner (e.g., ethereum)",
                placeholder="Enter GitHub Owner name",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_repo = gr.Textbox(
                label="GitHub Repository Name (e.g., go-ethereum)",
                placeholder="Enter GitHub Repository name",
                interactive=True,
                visible=False # Default hidden
            )
            start_date_picker = gr.DateTime(
                label="Start Date",
                value=datetime.now() - timedelta(days=config.DEFAULT_DAYS),
                interactive=True,
                type="datetime"
            )
            end_date_picker = gr.DateTime(
                label="End Date",
                value=datetime.now(),
                interactive=True,
                type="datetime"
            )
            
            gr.Markdown("### Strategy Backtesting Settings")
            strategy_choice = gr.CheckboxGroup(
                label="Select Backtesting Strategy(ies)",
                choices=["Simple Commit Threshold Strategy", "Commit SMA Strategy", "LLM Commit Analysis Strategy", "Simple SMA Strategy"], 
                value=[], # Default to no strategy selected
                interactive=True
            )
            buy_logic = gr.Radio(
                ["AND", "OR"],
                label="Buy Condition Logic",
                value="AND",
                interactive=True
            )
            buy_combination_mode = gr.Radio(
                ["同時", "非同時"],
                label="Buy Combination Mode (for AND logic only)",
                value="同時",
                interactive=True
            )
            sell_logic = gr.Radio(
                ["AND", "OR"],
                label="Sell Condition Logic",
                value="AND",
                interactive=True
            )
            sell_combination_mode = gr.Radio(
                ["同時", "非同時"],
                label="Sell Combination Mode (for AND logic only)",
                value="同時",
                interactive=True
            )
            buy_threshold_input = gr.Number(
                label="Buy Commit Threshold", 
                value=50,
                info="Buy when daily commits reach or exceed this value",
                interactive=True, 
                visible=False
            )
            buy_threshold_range_input = gr.Textbox(
                label="Buy Commit Threshold (Range/List)", 
                placeholder="e.g., 50,60,70 or 50-70-5",
                interactive=True, 
                visible=False
            )
            sell_threshold_input = gr.Number(
                label="Sell Commit Threshold", 
                value=10,
                info="Sell when daily commits fall below or equal to this value",
                interactive=True, 
                visible=False
            )
            sell_threshold_range_input = gr.Textbox(
                label="Sell Commit Threshold (Range/List)", 
                placeholder="e.g., 10,20,30 or 10-30-5",
                interactive=True, 
                visible=False
            )
            short_sma_period_input = gr.Number( 
                label="Short Commit SMA Period",
                value=5,
                info="Period (days) for calculating short-term Simple Moving Average",
                interactive=True,
                visible=False
            )
            short_sma_period_range_input = gr.Textbox( 
                label="Short Commit SMA Period (Range/List)",
                placeholder="e.g., 5,10,15 or 5-15-1",
                interactive=True,
                visible=False
            )
            long_sma_period_input = gr.Number( 
                label="Long Commit SMA Period",
                value=10,
                info="Period (days) for calculating long-term Simple Moving Average",
                interactive=True,
                visible=False
            )
            long_sma_period_range_input = gr.Textbox( 
                label="Long Commit SMA Period (Range/List)",
                placeholder="e.g., 10,20,30 or 10-30-5",
                interactive=True,
                visible=False
            )
            buy_score_threshold_input = gr.Number(
                label="Buy Score Threshold", 
                value=2,
                info="Buy when daily commit analysis score reaches or exceed this value",
                interactive=True, 
                visible=False
            )
            buy_score_threshold_range_input = gr.Textbox(
                label="Buy Score Threshold (Range/List)", 
                placeholder="e.g., 1,2,3 or 1-3-1",
                interactive=True, 
                visible=False
            )
            sell_score_threshold_input = gr.Number(
                label="Sell Score Threshold", 
                value=-2,
                info="Sell when daily commit analysis score falls below or equal to this value",
                interactive=True, 
                visible=False
            )
            sell_score_threshold_range_input = gr.Textbox(
                label="Sell Score Threshold (Range/List)", 
                placeholder="e.g., -1,-2,-3 or -1--3--1",
                interactive=True, 
                visible=False
            )
            sma1_period_input = gr.Number(
                label="SMA1 Period (shorter)",
                value=10,
                info="Period (days) for the shorter Simple Moving Average",
                interactive=True,
                visible=False
            )
            sma1_period_range_input = gr.Textbox(
                label="SMA1 Period (Range/List)",
                placeholder="e.g., 5,10,15 or 5-15-5",
                interactive=True,
                visible=False
            )
            sma2_period_input = gr.Number(
                label="SMA2 Period (longer)",
                value=30,
                info="Period (days) for the longer Simple Moving Average",
                interactive=True,
                visible=False
            )
            sma2_period_range_input = gr.Textbox(
                label="SMA2 Period (Range/List)",
                placeholder="e.g., 20,30,40 or 20-40-10",
                interactive=True,
                visible=False
            )

            ic_ir_prediction_horizon_input = gr.Number(
                label="IC/IR Prediction Horizon (N days)",
                value=7,
                info="Number of days into the future for IC/IR calculation (N)",
                interactive=True
            )
            ic_ir_prediction_horizon_range_input = gr.Textbox(
                label="IC/IR Prediction Horizon (N days) (Range/List)",
                placeholder="e.g., 5,7,10 or 5-15-1",
                interactive=True,
                visible=False
            )

            apply_commission_checkbox = gr.Checkbox(
                label="Apply Commissions to Strategy Return Curve",
                value=True,
                interactive=True,
                info="Toggle to see strategy return curve with/without commissions."
            )

            dynamic_updates_checkbox = gr.Checkbox(
                label="Enable Dynamic Updates",
                value=True,
                interactive=True,
                info="Show backtesting results in real-time. Disable for faster final results."
            )
            
            hyperparameter_search_mode = gr.Checkbox(
                label="Enable Hyperparameter Search Mode",
                value=False,
                interactive=True,
                info="If enabled, strategy parameters can be entered as ranges (e.g., '5,10,15' or '5-15-5')."
            )

            analyze_button = gr.Button("Analyze and Generate Charts")
            output_terminal_log = gr.Textbox(label="Backend Log", interactive=False, lines=5, autoscroll=True, max_lines=10)
        
        with gr.Column():
            output_plot_price_commits = gr.Plot(label="Price Trend vs. GitHub Commit Count (with Buy/Sell Points)")
            output_plot_returns = gr.Plot(label="Strategy Cumulative Return Curve")
            output_performance_metrics = gr.Textbox(label="Strategy Performance Metrics", interactive=False, lines=10, max_lines=20)
            output_message = gr.Textbox(label="Status/Message", interactive=False)
            output_trade_log = gr.Dataframe(
                headers=["時間", "幣種", "買進價格", "賣出價格", "買進時間", "賣出時間", "持有時間", "利潤", "利潤%數", "當前總資產", "當前總報酬率%數"],
                row_count=5,  # Display 5 rows at a time
                col_count=11, # Number of columns
                wrap=True,
                interactive=False,
                label="Trade Log"
            )
            output_hyperparameter_results = gr.Dataframe(
                headers=[
                    "組合", "總報酬率(%)", "年化報酬率(%)", "最大回撤(%)", "夏普比率", "索提諾比率", "勝率(%)", "總交易次數",
                    "IC", "IR",
                    "Simple Commit Threshold Params", "Commit SMA Params", "LLM Commit Analysis Params", "Simple SMA Params", "IC/IR Prediction Horizon"
                ],
                row_count=10, # Display more rows for results
                col_count=15, # Number of columns
                wrap=True,
                interactive=False,
                label="Hyperparameter Search Results",
                visible=False # Initially hidden
            )

    # Function to toggle visibility of manual input fields and pre-fill if predefined
    def toggle_manual_input_visibility_and_fill(choice):
        if choice == "Manual GitHub Repo & Binance Symbol":
            return (gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""))
        elif choice in config.PREDEFINED_CRYPTOS:
            crypto_info = config.PREDEFINED_CRYPTOS[choice]
            return (gr.update(visible=False, value=crypto_info["binance_symbol"]), 
                    gr.update(visible=False, value=crypto_info["github_owner"]),
                    gr.update(visible=False, value=crypto_info["github_repo"]))
        else: 
            return (gr.update(visible=False, value=choice), 
                    gr.update(visible=True, value=""), 
                    gr.update(visible=True, value="")) 

    crypto_choice.change(
        toggle_manual_input_visibility_and_fill,
        inputs=crypto_choice,
        outputs=[manual_coingecko_id, manual_github_owner, manual_github_repo]
    )

    # Add change event handlers for date pickers to ensure UI updates
    start_date_picker.change(lambda x: x, inputs=start_date_picker, outputs=start_date_picker)
    end_date_picker.change(lambda x: x, inputs=end_date_picker, outputs=end_date_picker)

    # Function to toggle visibility of strategy parameters and output displays
    def toggle_strategy_params_visibility(strategy_choice_values, is_hyperparameter_search_mode):
        # Simple Commit Threshold Strategy
        buy_thresh_single_vis = False
        buy_thresh_range_vis = False
        sell_thresh_single_vis = False
        sell_thresh_range_vis = False

        # Commit SMA Strategy
        short_sma_single_vis = False
        short_sma_range_vis = False
        long_sma_single_vis = False
        long_sma_range_vis = False

        # LLM Commit Analysis Strategy
        buy_score_single_vis = False
        buy_score_range_vis = False
        sell_score_single_vis = False
        sell_score_range_vis = False

        # Simple SMA Strategy
        sma1_period_single_vis = False
        sma1_period_range_vis = False
        sma2_period_single_vis = False
        sma2_period_range_vis = False

        # IC/IR Prediction Horizon
        ic_ir_single_vis = False
        ic_ir_range_vis = False

        # Output visibility
        plots_visible = not is_hyperparameter_search_mode
        trade_log_visible = not is_hyperparameter_search_mode
        hyperparameter_results_visible = is_hyperparameter_search_mode

        for strategy_choice_value in strategy_choice_values:
            if strategy_choice_value == "Simple Commit Threshold Strategy":
                if is_hyperparameter_search_mode:
                    buy_thresh_range_vis = True
                    sell_thresh_range_vis = True
                else:
                    buy_thresh_single_vis = True
                    sell_thresh_single_vis = True
            elif strategy_choice_value == "Commit SMA Strategy":
                if is_hyperparameter_search_mode:
                    short_sma_range_vis = True
                    long_sma_range_vis = True
                else:
                    short_sma_single_vis = True
                    long_sma_single_vis = True
            elif strategy_choice_value == "LLM Commit Analysis Strategy":
                if is_hyperparameter_search_mode:
                    buy_score_range_vis = True
                    sell_score_range_vis = True
                else:
                    buy_score_single_vis = True
                    sell_score_single_vis = True
            elif strategy_choice_value == "Simple SMA Strategy":
                if is_hyperparameter_search_mode:
                    sma1_period_range_vis = True
                    sma2_period_range_vis = True
                else:
                    sma1_period_single_vis = True
                    sma2_period_single_vis = True
        
        if is_hyperparameter_search_mode:
            ic_ir_range_vis = True
        else:
            ic_ir_single_vis = True

        return (gr.update(visible=buy_thresh_single_vis), gr.update(visible=buy_thresh_range_vis),
                gr.update(visible=sell_thresh_single_vis), gr.update(visible=sell_thresh_range_vis),
                gr.update(visible=short_sma_single_vis), gr.update(visible=short_sma_range_vis),
                gr.update(visible=long_sma_single_vis), gr.update(visible=long_sma_range_vis),
                gr.update(visible=buy_score_single_vis), gr.update(visible=buy_score_range_vis),
                gr.update(visible=sell_score_single_vis), gr.update(visible=sell_score_range_vis),
                gr.update(visible=sma1_period_single_vis), gr.update(visible=sma1_period_range_vis),
                gr.update(visible=sma2_period_single_vis), gr.update(visible=sma2_period_range_vis),
                gr.update(visible=ic_ir_single_vis), gr.update(visible=ic_ir_range_vis),
                gr.update(visible=plots_visible), gr.update(visible=plots_visible), gr.update(visible=plots_visible), gr.update(visible=trade_log_visible), gr.update(visible=hyperparameter_results_visible))

    strategy_choice.change(
        toggle_strategy_params_visibility,
        inputs=[strategy_choice, hyperparameter_search_mode],
        outputs=[buy_threshold_input, buy_threshold_range_input, 
                 sell_threshold_input, sell_threshold_range_input,
                 short_sma_period_input, short_sma_period_range_input, 
                 long_sma_period_input, long_sma_period_range_input,
                 buy_score_threshold_input, buy_score_threshold_range_input, 
                 sell_score_threshold_input, sell_score_threshold_range_input,
                 sma1_period_input, sma1_period_range_input, 
                 sma2_period_input, sma2_period_range_input,
                 ic_ir_prediction_horizon_input, ic_ir_prediction_horizon_range_input,
                 output_plot_price_commits, output_plot_returns, output_performance_metrics, output_trade_log, output_hyperparameter_results]
    )

    hyperparameter_search_mode.change(
        toggle_strategy_params_visibility,
        inputs=[strategy_choice, hyperparameter_search_mode],
        outputs=[buy_threshold_input, buy_threshold_range_input, 
                 sell_threshold_input, sell_threshold_range_input,
                 short_sma_period_input, short_sma_period_range_input, 
                 long_sma_period_input, long_sma_period_range_input,
                 buy_score_threshold_input, buy_score_threshold_range_input, 
                 sell_score_threshold_input, sell_score_threshold_range_input,
                 sma1_period_input, sma1_period_range_input, 
                 sma2_period_input, sma2_period_range_input,
                 ic_ir_prediction_horizon_input, ic_ir_prediction_horizon_range_input,
                 output_plot_price_commits, output_plot_returns, output_performance_metrics, output_trade_log, output_hyperparameter_results]
    )

    analyze_button.click(
        analyze_crypto_activity,
        inputs=[
            crypto_choice, manual_coingecko_id, manual_github_owner, manual_github_repo,
            start_date_picker, end_date_picker,
            strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode, 
            buy_threshold_input, buy_threshold_range_input, 
            sell_threshold_input, sell_threshold_range_input, 
            short_sma_period_input, short_sma_period_range_input, 
            long_sma_period_input, long_sma_period_range_input,
            buy_score_threshold_input, buy_score_threshold_range_input, 
            sell_score_threshold_input, sell_score_threshold_range_input,
            sma1_period_input, sma1_period_range_input, 
            sma2_period_input, sma2_period_range_input,
            ic_ir_prediction_horizon_input, ic_ir_prediction_horizon_range_input,
            apply_commission_checkbox,
            dynamic_updates_checkbox,
            hyperparameter_search_mode
        ],
        outputs=[
            output_plot_price_commits,
            output_plot_returns,
            output_performance_metrics,
            output_message,
            output_terminal_log,
            output_trade_log,
            output_hyperparameter_results
        ],
        show_progress='full'
    )

    def display_selected_hyperparameter_result(evt: gr.SelectData,
                                               crypto_selection, manual_binance_symbol, manual_owner, manual_repo,
                                               start_date_input, end_date_input,
                                               strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode,
                                               buy_threshold_input, sell_threshold_input,
                                               short_sma_period_input, long_sma_period_input,
                                               buy_score_threshold_input, sell_score_threshold_input,
                                               sma1_period_input, sma2_period_input,
                                               ic_ir_prediction_horizon_input,
                                               apply_commission_to_plot, enable_dynamic_updates,
                                               hyperparameter_results_data): # Added hyperparameter_results_data as input
        
        row_index = evt.index[0] # Get the row index from evt.index
        selected_row_data = hyperparameter_results_data.iloc[row_index] # Get the selected row data from the full data

        # Extract parameters from the selected row data
        # The order of columns in output_hyperparameter_results is:
        # "組合", "總報酬率(%)", "年化報酬率(%)", "最大回撤(%)", "夏普比率", "索提諾比率", "勝率(%)", "總交易次數",
        # "Simple Commit Threshold Params", "Commit SMA Params", "LLM Commit Analysis Params", "Simple SMA Params"
        
        # Default values for parameters (if not present in the selected strategy)
        current_buy_threshold = buy_threshold_input
        current_sell_threshold = sell_threshold_input
        current_short_sma = short_sma_period_input
        current_long_sma = long_sma_period_input
        current_buy_score = buy_score_threshold_input
        current_sell_score = sell_score_threshold_input
        current_sma1 = sma1_period_input
        current_sma2 = sma2_period_input
        current_ic_ir_prediction_horizon = ic_ir_prediction_horizon_input

        # Parse Simple Commit Threshold Params
        simple_commit_params_str = selected_row_data[10] # Index 10 for "Simple Commit Threshold Params"
        if simple_commit_params_str:
            parts = simple_commit_params_str.split(', ')
            for part in parts:
                if part.startswith("BuyThresh:"):
                    current_buy_threshold = int(part.split(': ')[1])
                elif part.startswith("SellThresh:"):
                    current_sell_threshold = int(part.split(': ')[1])

        # Parse Commit SMA Params
        commit_sma_params_str = selected_row_data[11] # Index 11 for "Commit SMA Params"
        if commit_sma_params_str:
            parts = commit_sma_params_str.split(', ')
            for part in parts:
                if part.startswith("ShortSMA:"):
                    current_short_sma = int(part.split(': ')[1])
                elif part.startswith("LongSMA:"):
                    current_long_sma = int(part.split(': ')[1])

        # Parse LLM Commit Analysis Params
        llm_params_str = selected_row_data[12] # Index 12 for "LLM Commit Analysis Params"
        if llm_params_str:
            parts = llm_params_str.split(', ')
            for part in parts:
                if part.startswith("BuyScore:"):
                    current_buy_score = int(part.split(': ')[1])
                elif part.startswith("SellScore:"):
                    current_sell_score = int(part.split(': ')[1])

        # Parse Simple SMA Params
        simple_sma_params_str = selected_row_data[13] # Index 13 for "Simple SMA Params"
        if simple_sma_params_str:
            parts = simple_sma_params_str.split(', ')
            for part in parts:
                if part.startswith("SMA1:"):
                    current_sma1 = int(part.split(': ')[1])
                elif part.startswith("SMA2:"):
                    current_sma2 = int(part.split(': ')[1])

        # Parse IC/IR Prediction Horizon
        ic_ir_prediction_horizon_str = selected_row_data[14] # Index 14 for "IC/IR Prediction Horizon"
        if ic_ir_prediction_horizon_str:
            try:
                current_ic_ir_prediction_horizon = int(ic_ir_prediction_horizon_str.split(': ')[1])
            except (ValueError, IndexError):
                pass # Keep default if parsing fails

        # Parse IC/IR Prediction Horizon
        ic_ir_prediction_horizon_str = selected_row_data[14] # Index 14 for "IC/IR Prediction Horizon"
        if ic_ir_prediction_horizon_str:
            try:
                current_ic_ir_prediction_horizon = int(ic_ir_prediction_horizon_str.split(': ')[1])
            except (ValueError, IndexError):
                pass # Keep default if parsing fails

        # Now call analyze_crypto_activity with the extracted parameters and hyperparameter_search_mode=False
        results_generator = analyze_crypto_activity(
            crypto_selection, manual_binance_symbol, manual_owner, manual_repo,
            start_date_input, end_date_input,
            strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode,
            current_buy_threshold, "", # Pass single value, empty string for range
            current_sell_threshold, "",
            current_short_sma, "",
            current_long_sma, "",
            current_buy_score, "",
            current_sell_score, "",
            current_sma1, "",
            current_sma2, "",
            current_ic_ir_prediction_horizon, "", # Pass single value, empty string for range
            apply_commission_to_plot, enable_dynamic_updates, False # hyperparameter_search_mode = False
        )
        
        # Iterate through the generator to get the final result
        final_results = None
        for result in results_generator:
            final_results = result
            
        fig1, fig2, performance_text, status_message, terminal_log, trades_info_formatted, _ = final_results
        
        print(f"DEBUG: Type of fig1: {type(fig1)}")
        print(f"DEBUG: Type of fig2: {type(fig2)}")
        # Make plots and trade log visible, hide hyperparameter results
        return (gr.update(value=fig1, visible=True), gr.update(value=fig2, visible=True), gr.update(value=performance_text, visible=True), gr.update(value=status_message), gr.update(value=terminal_log), gr.update(value=trades_info_formatted, visible=True), gr.update(visible=True))

    output_hyperparameter_results.select(
        display_selected_hyperparameter_result,
        inputs=[
            crypto_choice, manual_coingecko_id, manual_github_owner, manual_github_repo,
            start_date_picker, end_date_picker,
            strategy_choice, buy_logic, sell_logic, buy_combination_mode, sell_combination_mode,
            buy_threshold_input, sell_threshold_input,
            short_sma_period_input, long_sma_period_input,
            buy_score_threshold_input, sell_score_threshold_input,
            sma1_period_input, sma2_period_input,
            ic_ir_prediction_horizon_input,
            apply_commission_checkbox, dynamic_updates_checkbox,
            output_hyperparameter_results # Add the dataframe itself as an input
        ],
        outputs=[
            output_plot_price_commits,
            output_plot_returns,
            output_performance_metrics,
            output_message,
            output_terminal_log,
            output_trade_log,
            output_hyperparameter_results
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=10000)