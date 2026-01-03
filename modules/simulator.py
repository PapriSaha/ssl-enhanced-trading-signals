import pandas as pd
from backtesting import Backtest, Strategy


class SignalBandStrategy(Strategy):
    """
    Implements the trading logic for the `backtesting` library.
    It interprets the -1, 0, 1 signals from the ML model and executes trades.
    Includes Risk Management (Stop Loss / Take Profit).
    """

    # Define parameters (can be overridden)
    sl = 0.015  # 1.5% Stop Loss
    tp = 0.030  # 3.0% Take Profit (1:2 Risk/Reward)

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.last_signal = None

    def init(self):
        self.last_signal = 0

    def next(self):
        signal = self.data.Signal[-1]

        if signal == 0:
            return  # Hold

        if signal != self.last_signal:
            if self.position:
                self.position.close()

            price = self.data.Close[-1]

            if signal == 1:  # Buy
                # SL is below price, TP is above price
                sl_price = price * (1 - self.sl)
                tp_price = price * (1 + self.tp)
                self.buy(sl=sl_price, tp=tp_price)

            elif signal == -1:  # Sell
                # SL is above price, TP is below price
                sl_price = price * (1 + self.sl)
                tp_price = price * (1 - self.tp)
                self.sell(sl=sl_price, tp=tp_price)

            self.last_signal = signal


def infer_and_add_date(df, start_date="2000-01-01", candles_per_day=24):
    hours = 24 / candles_per_day
    freq = f"{int(hours)}H"

    df = df.copy()
    df["Date"] = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df


def run_backtesting_simulator(df, cash=10000, commission=0.002, plot=True):
    df = df.copy()  # 🔑 IMPORTANT

    if "Date" not in df.columns:
        raise ValueError("Please provide Date and Time included data for appropriate simulation.")

    # Always set Date as index for correct Sharpe/Sortino calculation
    if "Date" in df.columns:
        df.set_index('Date', inplace=True)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }

    df.rename(columns=rename_map, inplace=True)

    bt = Backtest(
        df,
        SignalBandStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True
    )

    stats = bt.run()

    if plot:
        bt.plot(resample=False)

    return stats
