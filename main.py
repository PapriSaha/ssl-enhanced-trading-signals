#!/usr/bin/env python
# coding: utf-8
"""
Trading Signal ML Pipeline - SSL with MA Crossover + Ensemble
"""

import os
import pandas as pd
import argparse  # For parsing command line arguments
import warnings

from modules.data_loader import DataLoader
from modules.feature_manager import FeatureGenerator
from modules.label_manager import LabelGenerator
from modules.model_trainer import EnsembleTrainer
from modules.evaluator import ModelEvaluator
from modules.strategy_simulator import StrategySimulator

warnings.filterwarnings('ignore')


class SSLEnsemblePipeline:
    """
    coordinate all the specialized modules into a coherent pipeline (Load -> Feature -> Label -> Train -> Test -> Save).
    """

    def __init__(self, train_path, test_path, forward_bars=5, threshold=0.005, n_features=50):
        # Store configuration parameters
        self.train_path = train_path
        self.test_path = test_path
        self.forward_bars = forward_bars  # Lookahead window
        self.threshold = threshold  # Profit target
        self.n_features = n_features  # Number of features to select

        # Initialize module instances
        # We create objects of each class to use their methods later
        self.data_loader = DataLoader()
        self.feature_gen = FeatureGenerator()
        self.label_gen = LabelGenerator()
        self.trainer = EnsembleTrainer()
        self.evaluator = ModelEvaluator()
        self.simulator = StrategySimulator()

        # Initialize placeholders for data
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self):
        """
        Executes the pipeline steps in order.
        """
        print("\n" + "=" * 80)
        print("TRADING SIGNAL ML PIPELINE - SSL + ENSEMBLE")
        print("=" * 80)

        self._load_data()
        self._generate_features_and_labels()
        self._train_model()
        self._test_model()
        self._run_simulation()
        self._save_model()

    def _load_data(self):
        """Step 1: Get the data from disk."""
        print("\n" + "=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)

        print("\n[1/2] Loading TRAINING data (EURUSD)...")
        # Use DataLoader to get cleaned dataframe
        self.train_data = self.data_loader.load_and_clean(self.train_path)

        print("\n[2/2] Loading TEST data (GBPUSD)...")
        self.test_data = self.data_loader.load_and_clean(self.test_path)

    def _generate_features_and_labels(self):
        """Step 2: Prepare inputs (Features) and outputs (Labels)."""
        print("\n" + "=" * 80)
        print("STEP 2: FEATURE ENGINEERING & LABEL GENERATION")
        print("=" * 80)

        print(f"\n[1/2] Processing TRAINING data...")
        print(f"  Forward bars: {self.forward_bars}, Profit threshold: {self.threshold:.3%}")

        # GENERATE LABELS (The 'Answer Key')
        # We calculate this first because we need the full dataset length before feature lagging
        self.y_train = self.label_gen.generate_future_labels(self.train_data, self.forward_bars, self.threshold)
        self.y_test = self.label_gen.generate_future_labels(self.test_data, self.forward_bars, self.threshold)

        # GENERATE FEATURES (The 'Question')
        print("\n  Creating domain-invariant features...")
        self.X_train = self.feature_gen.create_features(self.train_data)
        self.X_test = self.feature_gen.create_features(self.test_data)

        # NORMALIZE VOLATILITY (Cross-Domain Adaptation)
        print("  Applying volatility-adjusted normalization...")
        # Scale test features based on how volatile the test asset is compared to training asset
        self.X_test = self.feature_gen.normalize_volatility(
            self.train_data, self.X_test, self.test_data['Close'], self.X_test.columns
        )

        # ALIGN LENGTHS
        # Feature generation creates NaNs at start (lags).
        # Label generation creates NaNs at end (future lookahead).
        # We trim the end (forward_bars) because we don't know the future for the very last candles.
        self.X_train = self.X_train.iloc[:-self.forward_bars]
        self.X_test = self.X_test.iloc[:-self.forward_bars]
        # Trim labels to match features length
        self.y_train = self.y_train.iloc[:len(self.X_train)]
        self.y_test = self.y_test.iloc[:len(self.X_test)]

        print(f"\nTraining features generated: {self.X_train.shape}")

    def _train_model(self):
        """Step 3: Train the AI."""
        # Delegate to Trainer module which handles Selection -> SSL -> SMOTE -> Ensemble
        self.model, self.scaler, self.selected_features = self.trainer.train(
            self.X_train, self.y_train, self.n_features
        )

        # Sanity Check: Visualize what the model learned on training data
        self.simulator.visualize_training_signals(self.train_data, self.trainer.train_pred)

    def _test_model(self):
        """Step 4: Check performance on unseen data."""
        print("\n" + "=" * 80)
        print("STEP 4: MODEL TESTING (UNSEEN DATA)")
        print("=" * 80)

        # Prepare test data:
        # 1. Select only the features the model used
        X_test_sel = self.X_test[self.selected_features]
        # 2. Scale them using the SAME scaler from training (do not fit scaler on test data!)
        X_test_sc_array = self.scaler.transform(X_test_sel)
        self.X_test_sc = pd.DataFrame(X_test_sc_array, columns=self.selected_features, index=X_test_sel.index)

        # Evaluate performance metrics (Accuracy, F1, etc.)
        self.test_pred = self.evaluator.evaluate(self.model, self.X_test_sc, self.y_test, dataset_name="TEST")

    def _run_simulation(self):
        """Step 5: See if we would make money."""
        # Run simulation with strategies (includes Confidence Filtering)
        self.simulator.run_simulation(
            self.model, self.X_test_sc, self.test_data, self.test_pred
        )

    def _save_model(self):
        """Step 6: Save everything so we can trade live later."""
        config = {
            'n_features': self.n_features,
            'forward_bars': self.forward_bars,
            'threshold': self.threshold
        }
        self.trainer.save_artifacts(config=config)


if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward_bars", type=int, default=5, help="Candles to look ahead for target")
    parser.add_argument("--threshold", type=float, default=0.005, help="Minimum return to classify as Buy/Sell")
    parser.add_argument("--n_features", type=int, default=50, help="Number of top features to select")
    args = parser.parse_args()

    # Define Paths
    DATA_DIR = r"C:\Users\HP\OneDrive\Desktop\Foundation in Computer Programming\FINAL-PROJECTS\datasets"
    TRAIN = os.path.join(DATA_DIR, "Cleaned_Signal_EURUSD_for_training_635_635_60000.csv")
    TEST = os.path.join(DATA_DIR, "GBPUSD_H1_20140525_20251021.csv")

    # Initialize Pipeline
    pipeline = SSLEnsemblePipeline(
        TRAIN, TEST,
        forward_bars=args.forward_bars,
        threshold=args.threshold,
        n_features=args.n_features
    )

    # Run Pipeline
    pipeline.run()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)