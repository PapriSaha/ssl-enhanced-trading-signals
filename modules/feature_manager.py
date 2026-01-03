import pandas as pd
import numpy as np


class FeatureGenerator:
    """
    Handles feature engineering for financial time series data.
    Why: Raw prices (Open/Close) are not stationary and hard for ML to learn.
    We convert them into technical indicators (RSI, MACD, etc.) that represent market state.
    """

    def create_features(self, df):
        """
        Main function to create domain-invariant features.
        Why: 'Domain-invariant' means features that work regardless of the price level (e.g., $1.00 vs $1.50).
        We use percentages and oscillators instead of raw prices.
        """
        df = df.copy()  # Work on a copy
        eps = 1e-9  # Small epsilon to prevent DivisionByZero errors

        # 1. Log Returns
        # Why: Log returns are symmetric and additive, better for statistical analysis than simple % change.
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # Create lagged returns (Past values)
        # Why: The market has 'memory'. Validating if past 1, 2, 5 candles influence future.
        for lag in [1, 2, 3, 5, 10, 15, 20]:
            df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)

        # 2. Cumulative Returns ( Momentum over windows)
        for p in [3, 5, 10, 20]:
            df[f'cumret_{p}'] = df['log_ret'].rolling(p).sum()

        # 3. Volatility / Candle Shape
        # Why: Large bodies indicate strong conviction. Large shadows indicate rejection/uncertainty.
        df['range_pct'] = (df['High'] - df['Low']) / (df['Close'] + eps)  # Total intra-candle volatility
        df['body_pct'] = (df['Close'] - df['Open']) / (df['Close'] + eps)  # Directional strength

        # Shadows (Wicks)
        upper = df[['Open', 'Close']].max(axis=1)
        lower = df[['Open', 'Close']].min(axis=1)
        df['upper_shadow_pct'] = (df['High'] - upper) / (df['Close'] + eps)  # Selling pressure at top
        df['lower_shadow_pct'] = (lower - df['Low']) / (df['Close'] + eps)  # Buying pressure at bottom

        # Volatility over time (Standard Deviation of returns)
        for w in [5, 10, 20]:
            df[f'vol_{w}'] = df['log_ret'].rolling(w).std()

        # Volatility Ratio: Short-term vol vs Long-term vol
        # Why: Volatility squeeze or expansion often precedes big moves.
        df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + eps)

        # 4. RSI (Relative Strength Index) - Momentum Oscillator
        # Why: standard measure of overbought (>70) or oversold (<30) conditions.
        for p in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            df[f'rsi_{p}'] = 100 - 100 / (1 + gain / (loss + eps))

        # 5. Stochastic Oscillator
        # Why: Compares closing price to price range over time.
        for p in [7, 14, 21]:
            low_n = df['Low'].rolling(p).min()
            high_n = df['High'].rolling(p).max()
            df[f'stoch_k_{p}'] = 100 * (df['Close'] - low_n) / (high_n - low_n + eps)  # %K line
            df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()  # %D line (signal)

        # 6. Williams %R - Momentum
        # Why: Similar to Stochastic but scale is 0 to -100. Good for reversal detection.
        for p in [7, 14, 21]:
            high_n = df['High'].rolling(p).max()
            low_n = df['Low'].rolling(p).min()
            df[f'williams_r_{p}'] = -100 * (high_n - df['Close']) / (high_n - low_n + eps)

        # 7. MA Distances (Moving Average deviation)
        # Why: Measures how far price has extended from the mean (mean reversion potential).
        for w in [5, 10, 20, 50, 100, 200]:
            ma = df['Close'].rolling(w).mean()
            df[f'dist_sma_{w}'] = (df['Close'] - ma) / (ma + eps)

        # 8. Bollinger Bands
        # Why: Measures volatility and relative price level within standard deviation bands.
        for w in [10, 20, 30]:
            ma = df['Close'].rolling(w).mean()
            std = df['Close'].rolling(w).std()
            # %B: Where is price relative to bands? (0 = lower band, 1 = upper band)
            df[f'bb_percent_b_{w}'] = (df['Close'] - (ma - 2 * std)) / (4 * std + eps)
            # Bandwidth: How wide are the bands? (Volatility measure)
            df[f'bb_width_pct_{w}'] = (4 * std) / (ma + eps)

        # 9. ATR (Average True Range) - Volatility
        # Why: Best measure of absolute volatility (includes gaps).
        tr = np.maximum(df['High'] - df['Low'],
                        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
        for p in [7, 14, 21]:
            # Normalize ATR by price to make it percentage-based
            df[f'atr_pct_{p}'] = tr.rolling(p).mean() / (df['Close'] + eps)

        # 10. ADX (Average Directional Index) - Trend Strength
        # Why: Determines if market is trending or ranging.
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr + eps)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + eps)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + eps)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # 11. Price Position in Range
        # Why: Simple view of where we are relative to recent Low/Highs.
        for p in [5, 10, 20, 50]:
            high_n = df['High'].rolling(p).max()
            low_n = df['Low'].rolling(p).min()
            df[f'price_position_{p}'] = (df['Close'] - low_n) / (high_n - low_n + eps)

        # 12. Z-score
        # Why: Statistical measure of how unusual the current price is.
        for w in [20, 50, 100]:
            df[f'zscore_{w}'] = (df['Close'] - df['Close'].rolling(w).mean()) / (df['Close'].rolling(w).std() + eps)

        # 13. MACD (Moving Average Convergence Divergence)
        # Why: Classic trend-following momentum indicator.
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df['macd_norm'] = (macd - signal) / (df['Close'] + eps)  # Histogram
        df['ema_cross'] = (ema12 - ema26) / (df['Close'] + eps)  # Raw MACD line

        # 14. Momentum %
        for p in [5, 10, 20]:
            df[f'momentum_pct_{p}'] = (df['Close'] - df['Close'].shift(p)) / (df['Close'].shift(p) + eps)

        # 15. Volume Trends (if volume exists)
        # Why: Volume confirms price. High volume move is more significant.
        if 'Volume' in df.columns and df['Volume'].notna().sum() > 0 and (df['Volume'] > 0).any():
            for p in [5, 10, 20]:
                vol_ma = df['Volume'].rolling(p).mean() + eps
                df[f'vol_rel_{p}'] = df['Volume'] / vol_ma  # Relative Volume (RVOL)
            df['volume_trend'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + eps)

        # CLEANUP
        # Remove raw columns to only leave features for the model
        drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'time', 'Time',
                'spread', 'real_volume', 'Signal', 'Adj Close**']
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

        # Replace Infs/NaNs with 0 to prevent model crashes
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        return df

    def normalize_volatility(self, train_df, X_test, test_df_close, feature_cols):
        """
        Applies volatility-adjusted normalization for cross-domain transfer.
        Why: EURUSD might move 0.5% a day, while GBPUSD moves 0.8% a day.
        If we don't normalize, a 0.6% move looks 'huge' to EURUSD but 'normal' to GBPUSD.
        We scale the features so the model perceives the 'magnitude' correctly on the new pair.
        """
        # Calculate volatility (std dev of returns) for Train data
        train_vol = train_df['Close'].pct_change().std()

        # Calculate volatility for Test data
        test_vol = test_df_close.pct_change().std()

        # Calculate Ratio: "How much more volatile is Test vs Train?"
        vol_ratio = test_vol / train_vol

        print(f"  Train volatility: {train_vol:.6f}, Test volatility: {test_vol:.6f}, Ratio: {vol_ratio:.2f}")

        # Identify features that are volatility-dependent (contain 'vol' or 'atr')
        vol_features = [c for c in feature_cols if 'vol' in c.lower() or 'atr' in c.lower()]
        X_test_adj = X_test.copy()

        # Adjust those features by dividing by the ratio
        # Example: If GBP is 2x more volatile, we divide GBP ATR by 2 to make it look like EUR ATR.
        for col in vol_features:
            if col in X_test_adj.columns:
                X_test_adj[col] = X_test_adj[col] / vol_ratio

        return X_test_adj
