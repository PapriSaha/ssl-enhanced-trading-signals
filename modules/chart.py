import matplotlib.pyplot as plt


def generate_signal_plot(data_plot, val_limit=5000000):
    df = data_plot.iloc[:val_limit, :].copy()

    # Auto-detect close column
    close_col = None
    for col in df.columns:
        if col.lower() == "close":
            close_col = col
            break

    if close_col is None:
        raise KeyError("No 'Close' or 'close' column found in dataframe.")

    # Plot Close Prices and signals
    plt.figure(figsize=(10, 7))
    plt.plot(df.index, df[close_col], c='black', alpha=0.7,
             label='Close Price', linewidth=0.5)

    # Combine Buy and Sell Signals
    if "Signal" not in df.columns:
        raise KeyError("No 'Signal' column found in dataframe.")

    signals = df[df['Signal'] != 0]

    # Plot Sell Signals
    sell_signals = signals[signals['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals[close_col],
                c='red', label='Sell Signal', marker='o', s=14)

    # Plot Buy Signals
    buy_signals = signals[signals['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals[close_col],
                c='blue', label='Buy Signal', marker='o', s=14)

    # Chart Customization
    plt.title('Buy/Sell Signals on Close Price', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title='Signal Type')
    plt.tight_layout()
    plt.show()