import numpy as np
import pandas as pd
from collections import Counter
from modules.simulator import run_backtesting_simulator
from modules.chart import generate_signal_plot


class StrategySimulator:
    """
    Handles backtesting simulation and visualization.
    Why: ML metrics (Accuracy) don't equate to Money. We need to Simulate the strategy
    with trading rules (spread, commission, risk management) to see if it's profitable.
    """

    def run_simulation(self, model, X_test_sc, test_data, test_pred, buy_threshold=0.55, sell_threshold=0.55):
        """
        Runs the simulation with confidence filtering.
        Args:
            buy_threshold: Probability% required to execute a BUY.
            sell_threshold: Probability% required to execute a SELL.
        """
        print("\n" + "=" * 80)
        print("STEP 5: BACKTESTING SIMULATION")
        print("=" * 80)

        # [1/4] Smart signal filtering
        print("\n[1/4] Applying smart signal filtering for profitable trading...")

        # Get Probability Distribution from the model (e.g., [0.1, 0.8, 0.1] -> 80% Buy)
        test_proba = model.predict_proba(X_test_sc)

        # Get the Max Probability (Confidence) for each prediction
        # axis=1 means check across columns (classes) per row
        test_confidence = np.max(test_proba, axis=1)

        # Start with the raw predictions
        filtered_pred = test_pred.copy()

        # Apply Threshold Logic
        # If model Says BUY but confidence is only 51%, it's risky (coin flip).
        # We process each prediction one by one.
        for i in range(len(filtered_pred)):
            if filtered_pred[i] == 1:  # Proposed BUY
                if test_confidence[i] < buy_threshold:
                    filtered_pred[i] = 0  # Downgrade to HOLD if low confidence
            elif filtered_pred[i] == 2:  # Proposed SELL
                if test_confidence[i] < sell_threshold:
                    filtered_pred[i] = 0  # Downgrade to HOLD if low confidence

        # Report stats
        signal_counts = Counter(filtered_pred)
        print(f"  BUY threshold: {buy_threshold}, SELL threshold: {sell_threshold}")
        print(f"  Filtered signals: BUY={signal_counts[1]}, SELL={signal_counts[2]}, HOLD={signal_counts[0]}")

        # [2/4] Visualize Raw Predictions (Before Filtering)
        print("\n[2/4] Generating test signal visualization (Raw Model Predictions)...")
        try:
            # Map back to -1, 0, 1 for plotting
            raw_signals = pd.Series(test_pred).map({2: -1, 1: 1, 0: 0})

            # Create a subset dataframe for visualization
            viz_df = test_data.copy().iloc[:len(raw_signals)]
            viz_df['Signal'] = raw_signals.values
            if 'Date' in viz_df.columns:
                viz_df.set_index('Date', inplace=True)

            # Call helper to plot graph
            print("  -> Displaying Raw Signals Plot (Close Chart)...")
            generate_signal_plot(viz_df, val_limit=5000)
            print("  Test signal plot displayed successfully.")
        except Exception as e:
            print(f"    Visualization note: {e}")

        # [3/4] Backtest Simulation (Using FILTERED Signals)
        # Why: We trade the filtered (high quality) signals, not the raw ones.
        print("\n[3/4] Running backtest simulation (Using High-Confidence Signals)...")
        try:
            # Map filtered predictions to -1, 0, 1
            filtered_signals = pd.Series(filtered_pred).map({2: -1, 1: 1, 0: 0})

            # Prepare data for backtester
            sim_df = test_data.copy().iloc[:len(filtered_signals)]
            sim_df['Signal'] = filtered_signals.values  # Inject signals into dataframe

            # --- ADDED: Visualize Filtered Signals using chart.py style ---
            print("  -> Displaying Filtered Signals Plot (Backtesting Signals)...")
            try:
                # Use the same chart.py logic but with filtered signals
                filter_viz_df = sim_df.copy()
                if 'Date' in filter_viz_df.columns:
                    filter_viz_df.set_index('Date', inplace=True)

                # Plot the filtered signals
                generate_signal_plot(filter_viz_df, val_limit=5000)
                print("  Filtered signal plot displayed successfully.")
            except Exception as viz_e:
                print(f"  Filtered visualization note: {viz_e}")
            # -------------------------------------------------------------

            # Run Backtest (External Module)
            # Cash=100M (Forex standard lots), Commission=0.03% (Spread+Comm)
            stats = run_backtesting_simulator(sim_df, cash=100_000_000, commission=0.0003, plot=True)
            self._print_stats(stats)
        except Exception as e:
            print(f"  Backtesting note: {e}")

    def _print_stats(self, stats):
        """Helper to print backtest metrics nicely."""
        print("\n" + "=" * 80)
        print("BACKTESTING RESULTS")
        print("=" * 80)
        key_metrics = ['Return [%]', 'Win Rate [%]', 'Profit Factor', '# Trades',
                       'Sharpe Ratio', 'Sortino Ratio', 'Max. Drawdown [%]', 'Avg. Trade [%]',
                       'Best Trade [%]', 'Worst Trade [%]']

        for metric in key_metrics:
            if metric in stats.index:
                val = stats[metric]
                if pd.isna(val):
                    print(f"  {metric:.<35} N/A")
                elif isinstance(val, float):
                    print(f"  {metric:.<35} {val:.4f}")
                else:
                    print(f"  {metric:.<35} {val}")

    def visualize_training_signals(self, train_data, train_pred):
        """
        Visualizes signals on training data.
        Why: Sanity check. If training signals look random, the model isn't learning.
        """
        print("\n[4/4] Generating training signal visualization...")
        try:
            train_pred_signals = pd.Series(train_pred).map({2: -1, 1: 1, 0: 0})
            train_viz_df = train_data.copy().iloc[:len(train_pred_signals)]
            train_viz_df['Signal'] = train_pred_signals.values
            if 'Date' in train_viz_df.columns:
                train_viz_df.set_index('Date', inplace=True)
            generate_signal_plot(train_viz_df, val_limit=3000)
            print("  Training signal plot displayed successfully")
        except Exception as e:
            print(f"    Training visualization note: {e}")
