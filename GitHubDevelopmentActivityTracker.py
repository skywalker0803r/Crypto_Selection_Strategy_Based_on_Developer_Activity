import gradio as gr
from datetime import datetime, timedelta

import config
from data_fetcher import get_binance_trading_pairs
from app_logic import analyze_crypto_activity

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
                choices=["No Strategy", "Simple Commit Threshold Strategy", "Commit SMA Strategy", "LLM Commit Analysis Strategy"], 
                value=["No Strategy"], # Default to No Strategy
                interactive=True
            )
            buy_logic = gr.Radio(
                ["AND", "OR"],
                label="Buy Condition Logic",
                value="AND",
                interactive=True
            )
            sell_logic = gr.Radio(
                ["AND", "OR"],
                label="Sell Condition Logic",
                value="AND",
                interactive=True
            )
            buy_threshold_input = gr.Number(
                label="Buy Commit Threshold", 
                value=50,
                info="Buy when daily commits reach or exceed this value",
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
            short_sma_period_input = gr.Number( 
                label="Short Commit SMA Period",
                value=5,
                info="Period (days) for calculating short-term Simple Moving Average",
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
            buy_score_threshold_input = gr.Number(
                label="Buy Score Threshold", 
                value=2,
                info="Buy when daily commit analysis score reaches or exceed this value",
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

            analyze_button = gr.Button("Analyze and Generate Charts")
            output_terminal_log = gr.Textbox(label="Backend Log", interactive=False, lines=5, autoscroll=True, max_lines=10)
        
        with gr.Column():
            output_plot_price_commits = gr.Plot(label="Price Trend vs. GitHub Commit Count (with Buy/Sell Points)")
            output_plot_returns = gr.Plot(label="Strategy Cumulative Return Curve")
            output_performance_metrics = gr.Textbox(label="Strategy Performance Metrics", interactive=False)
            output_message = gr.Textbox(label="Status/Message", interactive=False)
            output_trade_log = gr.Dataframe(
                headers=["時間", "幣種", "買進價格", "賣出價格", "買進時間", "賣出時間", "持有時間", "利潤", "利潤%數", "當前總資產", "當前總報酬率%數"],
                row_count=5,  # Display 5 rows at a time
                col_count=11, # Number of columns
                wrap=True,
                interactive=False,
                label="Trade Log"
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

    # Function to toggle visibility of strategy parameters
    def toggle_strategy_params_visibility(strategy_choice_values):
        buy_thresh_vis = False
        sell_thresh_vis = False
        short_sma_vis = False
        long_sma_vis = False
        buy_score_vis = False
        sell_score_vis = False

        for strategy_choice_value in strategy_choice_values:
            if strategy_choice_value == "Simple Commit Threshold Strategy":
                buy_thresh_vis = True
                sell_thresh_vis = True
            elif strategy_choice_value == "Commit SMA Strategy":
                short_sma_vis = True
                long_sma_vis = True
            elif strategy_choice_value == "LLM Commit Analysis Strategy":
                buy_score_vis = True
                sell_score_vis = True
        
        return (gr.update(visible=buy_thresh_vis), 
                gr.update(visible=sell_thresh_vis), 
                gr.update(visible=short_sma_vis), 
                gr.update(visible=long_sma_vis),
                gr.update(visible=buy_score_vis),
                gr.update(visible=sell_score_vis))

    strategy_choice.change(
        toggle_strategy_params_visibility,
        inputs=strategy_choice,
        outputs=[buy_threshold_input, sell_threshold_input, short_sma_period_input, long_sma_period_input, buy_score_threshold_input, sell_score_threshold_input]
    )

    analyze_button.click(
        analyze_crypto_activity,
        inputs=[
            crypto_choice, manual_coingecko_id, manual_github_owner, manual_github_repo,
            start_date_picker, end_date_picker,
            strategy_choice, buy_logic, sell_logic, buy_threshold_input, sell_threshold_input, 
            short_sma_period_input, long_sma_period_input,
            buy_score_threshold_input, sell_score_threshold_input,
            apply_commission_checkbox,
            dynamic_updates_checkbox
        ],
        outputs=[
            output_plot_price_commits,
            output_plot_returns,
            output_performance_metrics,
            output_message,
            output_terminal_log,
            output_trade_log
        ],
        show_progress='full'
    )


demo.launch()