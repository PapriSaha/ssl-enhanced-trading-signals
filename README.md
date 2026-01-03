# Semi-Supervised Learning for Signal Enhancement and Market Prediction

This project is a comprehensive Machine Learning pipeline designed for generating trading signals (Buy/Sell/Hold) from financial time-series data. It specifically addresses the challenges of financial data, such as low signal-to-noise ratio and non-stationarity, by employing **Semi-Supervised Learning (SSL)** and **Ensemble Classification**.

The system is built using a modular, Object-Oriented architecture, implementing the **Facade Pattern** to orchestrate the workflow from data ingestion to backtesting simulation. The primary goal is to achieve domain-invariant performance, allowing models trained on one asset (e.g., EURUSD) to generalize effectively to others (e.g., GBPUSD).

## Key Features

- **Semi-Supervised Learning**: Uses `LabelPropagation` to leverage unlabeled data and smooth decision boundaries, improving generalization.
- **Ensemble Architecture**: Combines **LightGBM**, **XGBoost**, and **Random Forest** in a pure Soft Voting ensemble to maximize prediction stability.
- **Domain-Invariant Feature Engineering**: Generates 50+ technical indicators (RSI, MACD, Bollinger Bands, ATR, Volatility ratios) that rely on relative changes rather than absolute price levels.
- **Volatility-Adjusted Normalization**: Dynamically scales test data characteristics to match the training domain's volatility profile, enabling cross-asset transferability.
- **Smart Signal Filtering**: Post-processing logic that filters signals based on model confidence probabilities to reduce false positives.
- **Triple Barrier Labeling**: Implements advanced labeling logic considering fixed profit horizons and stop-loss mechanisms (simulated via forward windows).
- **Integrated Backtesting**: Includes a custom simulation engine and integration with the `backtesting` library to validate strategy profitability (Sharpe, Sortino, Drawdown).

## Directory Structure

```
├── main.py                  # Main entry point
├── README.md                # Project Documentation
├── modules/                 # Core Logic Modules
│   ├── data_loader.py       # Data Loading & Cleaning
│   ├── feature_manager.py   # Feature Engineering & Normalization
│   ├── label_manager.py     # Target Generation (Labeling)
│   ├── model_trainer.py     # SSL & Ensemble Training Pipeline
│   ├── evaluator.py         # Performance Metrics & Visualization
│   ├── simulator.py         # Backtesting Logic (Strategy Class)
│   └── strategy_simulator.py # Simulation Wrapper & Stats
├── datasets/                # Input CSV/Excel Files
└── models/                  # Saved Models & Artifacts
```

## Installation

### Prerequisites

- Python 3.8 or higher

### Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn backtesting openpyxl
```

**Key Libraries:**

- `pandas`, `numpy`: Data manipulation.
- `scikit-learn`: Random Forest, Label Propagation, Metrics, Scaling.
- `xgboost`, `lightgbm`: Gradient Boosting classifiers.
- `imbalanced-learn`: SMOTE for handling class imbalance.
- `backtesting`: Framework for strategy simulation.

## Usage

The main script is `main.py`. It requires training data (e.g., EURUSD) and testing data (e.g., GBPUSD).

### Running the Pipeline

```bash
python main.py --forward_bars 5 --threshold 0.005 --n_features 50
```

### Command Line Arguments

- `--forward_bars`: Number of candles to look ahead for the target label (default: `5`).
- `--threshold`: Minimum profit return (Log Return) required to classify a move as Buy/Sell (default: `0.005`).
- `--n_features`: Number of top features to select based on importance (default: `50`).

### Code Example

```python
# Initialize the Pipeline
pipeline = SSLEnsemblePipeline(
    train_path="datasets/Cleaned_Signal_EURUSD_for_training_635_635_60000.csv",
    test_path="datasets/GBPUSD_H1_20140525_20251021.csv",
    forward_bars=5,
    threshold=0.005,
    n_features=50
)

# Run the full workflow
pipeline.run()
```

## Modules Description

### 1. `data_loader.py`

Handles loading of CSV/Excel files and standardizes column names (Open, High, Low, Close, Volume). It also parses and fixes Date formats to ensure chronological integrity.

### 2. `feature_manager.py`

It creates domain-invariant features including:

- Log Returns & Momentum (Lags, Cumulative Returns)
- Oscillators (RSI, Stochastic, Williams %R)
- Volatility Measures (ATR, Bollinger Bands, Standard Deviation)
- Trend Indicators (ADX, MACD)
- **Volatility Normalization**: A critical function that adjusts the scale of features in the test set based on the volatility ratio between the training and testing assets.

### 3. `label_manager.py`

It looks into the future (`forward_bars`) to determine if the price will hit the profit `threshold` without hitting a stop-loss (implied by the net return check). Generates labels: `1` (Buy), `-1` (Sell), `0` (Hold).

### 4. `model_trainer.py`

It follows a 4-step process:

1.  **Feature Selection**: Uses a Random Forest to identify the most predictive features.
2.  **SSL (Semi-Supervised Learning)**: Uses `LabelPropagation` to infer labels for a subset of the data, smoothing the decision manifold.
3.  **Class Balancing**: Applies `SMOTE` with a custom strategy to oversample minority classes (Buy/Sell) to match the majority class (Hold).
4.  **Ensemble Training**: Trains a Soft Voting Classifier consisting of **LGBM**, **XGBoost**, and **Random Forest**.

### 5. `evaluator.py` & `strategy_simulator.py`

- **Evaluator**: Calculates technical ML metrics (Accuracy, Precision, Recall, F1-Score) and generates Confidence vs. Accuracy plots.
- **Simulator**: Converts model predictions into trades. It applies a "Confidence Filter" (e.g., only trade if model confidence > 55%) and runs a full backtest to report Sharpe Ratio, Max Drawdown, and Profit Factor.

## License

This project is available for academic and educational purposes.
