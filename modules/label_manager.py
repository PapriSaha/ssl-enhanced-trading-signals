import pandas as pd  # Core library for data manipulation


class LabelGenerator:
    """
    Handles label generation logic for trading signals.
    Defines 'What is a BUY?' and 'What is a SELL?' based on future or technical rules.
    """

    def generate_ma_crossover_signals(self, df, fast=10, slow=30):
        """
        Generate signals using Moving Average (MA) crossover.
        Why: This is a robust baseline strategy. Fast MA crossing slow MA indicates trend change.
        """
        df = df.copy()  # Create a copy to avoid Modifying the original dataframe

        # Calculate Fast Moving Average (e.g., 10-period)
        ma_fast = df['Close'].rolling(fast).mean()

        # Calculate Slow Moving Average (e.g., 30-period)
        ma_slow = df['Close'].rolling(slow).mean()

        # Initialize signals column with 0 (HOLD)
        signals = pd.Series(0, index=df.index)

        # BUY Signal Logic:
        # 1. Current Fast MA is higher than Slow MA (ma_fast > ma_slow)
        # 2. Previous Fast MA was LOWER or EQUAL to Previous Slow MA (ma_fast.shift(1) <= ...)
        # Why: This detects the exact moment of the 'crossover' upwards.
        buy = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        signals[buy] = 1  # Set label to 1 (BUY)

        # SELL Signal Logic:
        # 1. Current Fast MA is lower than Slow MA
        # 2. Previous Fast MA was HIGHER or EQUAL
        # Why: This detects the crossover downwards (start of downtrend).
        sell = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
        signals[sell] = -1  # Set label to -1 (SELL)

        return signals

    def generate_future_labels(self, df, forward_bars=5, threshold=0.002):
        """
        Generate labels based on future price movement (Look-Ahead).
        Why: Supervised learning needs 'Ground Truth'. We look into the future to see
        if the price actually went up or down, then teach the model to predict that.
        """
        # Get the price 'forward_bars' into the future (e.g., 5 candles ahead)
        future = df['Close'].shift(-forward_bars)

        # Calculate percentage return: (Future - Current) / Current
        # Why: We care about % change, not absolute price difference.
        ret = (future - df['Close']) / df['Close']

        # Initialize labels with 0 (HOLD)
        labels = pd.Series(0, index=df.index)

        # If return is positive and greater than threshold (e.g., > 0.2% profit)
        # Why: We only want to trade if the profit justifies the risk/spread.
        labels[ret > threshold] = 1  # Class 1: BUY

        # If return is negative and drop is larger than threshold (e.g., < -0.2%)
        # Why: Detect significant price drops for short selling.
        labels[ret < -threshold] = -1  # Class -1: SELL

        return labels
