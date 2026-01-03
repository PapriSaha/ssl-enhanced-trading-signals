import os
import joblib
import pandas as pd
import numpy as np
from collections import Counter

# Import Machine Learning Libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.semi_supervised import LabelPropagation
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class EnsembleTrainer:
    """
    Handles model creation, training, and saving.
    Includes Semi-Supervised Learning (SSL) and Ensemble methods.
    This encapsulates the entire ML workflow (Selection -> SSL -> Balancing -> Ensemble).
    """

    def __init__(self):
        # Initialize state variables
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.X_train_sc = None
        self.train_pred = None

    def train(self, X_train, y_train, n_features_to_select=70):
        """
        Runs the full training pipeline using a structured approach.
        Args:
            X_train: Feature matrix
            y_train: Target labels (-1, 0, 1)
            n_features_to_select: How many top features to keep
        """
        print("\n" + "=" * 80)
        print("STEP 3: MODEL TRAINING")
        print("=" * 80)

        # Map labels to non-negative integers for Scikit-Learn compatibility
        # -1 (SELL) -> 2
        #  0 (HOLD) -> 0
        #  1 (BUY)  -> 1
        y_train_map = y_train.map({-1: 2, 0: 0, 1: 1})

        # --- 1. Feature Selection ---
        print("\n[1/4] Selecting best features...")
        # Why: Using a Random Forest to find 'Feature Importance'.
        # We want to remove noisy columns that don't help prediction.
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train_map)

        # Create a dataframe of feature importance
        imp = pd.DataFrame({'feature': X_train.columns, 'importance': rf.feature_importances_})
        imp = imp.sort_values('importance', ascending=False)

        # Select top N features
        self.selected_features = imp.head(n_features_to_select)['feature'].tolist()

        print("\n=== Feature Importance (Top 15) ===")
        print(imp.head(15).to_string(index=False))

        # Filter dataset to only use selected features
        X_train_sel = X_train[self.selected_features]

        # --- 2. Scaling ---
        print("\n[2/4] Preparing features...")
        # Why: ML models (like SVM/Neural Nets) need scaled data.
        # Even trees benefit from stable numerical ranges.
        self.scaler = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Fill NaNs with median
            ('scaler', StandardScaler())  # Scale to mean=0, std=1
        ])

        # Fit and Transform training data
        X_train_sc_array = self.scaler.fit_transform(X_train_sel)
        # Convert back to DataFrame to keep column names
        self.X_train_sc = pd.DataFrame(X_train_sc_array, columns=self.selected_features, index=X_train_sel.index)

        print(f"Features prepared: {self.X_train_sc.shape}")
        print(f"  Class distribution: {Counter(y_train_map)}")

        # --- 3. Semi-Supervised Learning (SSL) ---
        print("\n[3/4] Applying Semi-Supervised Learning...")
        # Why: We want to smooth the decision boundaries.
        # We pretend some data is 'unknown' and let the algorithm fill it in based on neighbors.
        # This helps Generalized performance.

        # Create a copy of labels
        y_ssl = y_train_map.values.copy()

        # Masking: Randomly hide 15% of labels (set to -1)
        np.random.seed(42)
        mask = np.random.rand(len(y_ssl)) < 0.15
        y_ssl[mask] = -1  # -1 is the code for 'Unlabeled' in LabelPropagation

        print(f"  Labeled samples: {(y_ssl != -1).sum()}")
        print(f"  Unlabeled samples: {(y_ssl == -1).sum()}")

        # Train LabelPropagation
        lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=100, n_jobs=-1)
        lp.fit(self.X_train_sc.values, y_ssl)

        # Get the 'transduction': The refined labels (filled in)
        y_propagated = lp.transduction_
        print(f"  After propagation: {Counter(y_propagated)}")

        # Split a validation set (20%) from the training data for internal checking
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_sc, y_propagated, test_size=0.2, stratify=y_propagated, random_state=42
        )

        # --- 4. Class Balancing (SMOTE) ---
        print("\n  Applying SMOTE with aggressive oversampling for BUY/SELL...")
        # Why: Market data is 90% HOLD. We need to create synthetic BUY/SELLs so the model learns them.

        class_counts = Counter(y_tr)
        majority_count = max(class_counts.values())  # Count of HOLD class

        # Strategy: Make minority classes equal to the majority class
        sampling_strategy = {}
        for cls, count in class_counts.items():
            if count < majority_count:
                sampling_strategy[cls] = majority_count  # Boost minority to match majority
            else:
                sampling_strategy[cls] = count  # Keep majority as is

        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=7)
        X_balanced, y_balanced = smote.fit_resample(X_tr, y_tr)
        print(f"  After SMOTE: {Counter(y_balanced)}")

        # --- 5. Ensemble Training ---
        print("\n[4/4] Training Ensemble (LightGBM + XGBoost + RF)...")
        # Combining different models reduces variance and improves accuracy.
        # 'Wisdom of crowds'.

        # Model 1: LightGBM (Gradient Boosting)
        lgbm = lgb.LGBMClassifier(
            n_estimators=1300, max_depth=10, learning_rate=0.02,
            num_leaves=60, min_child_samples=20,  # Controls overfitting
            subsample=0.85, colsample_bytree=0.85,  # Use subset of rows/cols
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        )

        # Model 2: XGBoost (Gradient Boosting)
        xgb_model = XGBClassifier(
            n_estimators=1100, max_depth=9, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,  # L1/L2 regularization
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0
        )

        # Model 3: Random Forest (Bagging)
        rf_model = RandomForestClassifier(
            n_estimators=1000, max_depth=14, min_samples_split=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        )

        # Voting Classifier: The Ensemble
        # voting='soft' means we average the Probabilities, not just the labels.
        self.model = VotingClassifier(
            estimators=[('lgbm', lgbm), ('xgb', xgb_model), ('rf', rf_model)],
            voting='soft', n_jobs=1
        )

        # Train the ensemble on the Balanced dataset
        self.model.fit(X_balanced, y_balanced)

        # Validation Results
        val_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"\nValidation Accuracy (20% held-out): {val_acc:.2%}")

        # Store predictions for the full training set (for visualization later)
        self.train_pred = self.model.predict(self.X_train_sc)

        # Training Results
        train_acc = accuracy_score(y_train_map, self.train_pred)

        print("\n" + "=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print(f"Training Accuracy: {train_acc:.2%}")

        print("\nTraining Classification Report:")
        print(classification_report(y_train_map, self.train_pred,
                                    target_names=['HOLD (0)', 'BUY (1)', 'SELL (2)'], digits=3))

        print("\nTraining Confusion Matrix:")
        cm = confusion_matrix(y_train_map, self.train_pred)
        print(pd.DataFrame(cm, index=['True HOLD', 'True BUY', 'True SELL'],
                           columns=['Pred HOLD', 'Pred BUY', 'Pred SELL']))

        return self.model, self.scaler, self.selected_features

    def save_artifacts(self, path="models", config=None):
        """Saves models and configuration to disk."""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING MODEL")
        print("=" * 80)

        os.makedirs(path, exist_ok=True)
        # using joblib to serialize objects
        joblib.dump(self.scaler, os.path.join(path, "scaler_pipeline.pkl"))
        joblib.dump(self.selected_features, os.path.join(path, "selected_features.pkl"))
        joblib.dump(self.model, os.path.join(path, "ssl_ensemble_model.pkl"))

        if config:
            config['selected_features'] = self.selected_features
            config['model_type'] = 'SSL + Ensemble (LightGBM + XGBoost + RF)'
            joblib.dump(config, os.path.join(path, "model_config.pkl"))

        print(f"Model artifacts saved to ./{path}/")
