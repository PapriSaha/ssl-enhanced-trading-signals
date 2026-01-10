import pandas as pd
from backtesting import Backtest, Strategy


class SignalBandStrategy(Strategy):
    
    sl = 0.011  # 1.1% Stop Loss
    tp = 0.007  # 0.7% Take Profit

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.last_signal = None

    def init(self):
        self.last_signal = 0

    def next(self):
        signal = self.data.Signal[-1]

        if signal == 0:
            return

        if signal != self.last_signal:
            if self.position:
                self.position.close()

            price = self.data.Close[-1]

            if signal == 1:  # Buy
                self.buy(sl=price * (1 - self.sl), tp=price * (1 + self.tp))
            elif signal == -1:  # Sell
                self.sell(sl=price * (1 + self.sl), tp=price * (1 - self.tp))

            self.last_signal = signal


def infer_and_add_date(df, start_date="2000-01-01", candles_per_day=24):
    hours = 24 / candles_per_day
    freq = f"{int(hours)}H"
    df = df.copy()
    df["Date"] = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df


def run_backtesting_simulator(df, cash=10000, commission=0.002, plot=True):
    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError("Date column required.")

    if "Date" in df.columns:
        df.set_index('Date', inplace=True)

    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df.rename(columns=rename_map, inplace=True)

    bt = Backtest(df, SignalBandStrategy, cash=cash, commission=commission, exclusive_orders=True)
    stats = bt.run()

    if plot:
        bt.plot(resample=False)

    return stats
