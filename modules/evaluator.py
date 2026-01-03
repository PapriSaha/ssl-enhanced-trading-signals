import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelEvaluator:
    """
    Handles evaluation of model performance on test data.
    We need to quantify how good the model is using standard metrics.
    """

    def evaluate(self, model, X_test_sc, y_test, dataset_name="TEST"):
        """
        Evaluates the model on the provided data and labels.
        """
        print("\n" + "=" * 80)
        print(f"{dataset_name}ING RESULTS")
        print("=" * 80)

        # Map labels to 0, 1, 2 to match Model Output
        y_test_map = y_test.map({-1: 2, 0: 0, 1: 1})

        # Generate Predictions
        y_pred = model.predict(X_test_sc)

        # 1. Overall Accuracy
        # Why: Basic check. If < 33%, it's worse than random guessing.
        acc = accuracy_score(y_test_map, y_pred)
        print(f"{dataset_name} Accuracy: {acc:.2%}")

        # 2. Classification Report
        # Why: Accuracy is misleading in imbalanced data.
        # Precision/Recall per class is crucial (e.g., Do we catch the BUYs?)
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(y_test_map, y_pred,
                                    target_names=['HOLD (0)', 'BUY (1)', 'SELL (2)'], digits=3, zero_division=0))

        # 3. Confusion Matrix
        # Why: Visualizes exactly where the model is confused (e.g., Predicting BUY when it was actually SELL).
        print(f"\n{dataset_name} Confusion Matrix:")
        cm = confusion_matrix(y_test_map, y_pred)
        print(pd.DataFrame(cm, index=['True HOLD', 'True BUY', 'True SELL'],
                           columns=['Pred HOLD', 'Pred BUY', 'Pred SELL']))

        # 4. Per-Class Analysis
        self.analyze_per_class(y_test_map, y_pred)

        return y_pred

    def analyze_per_class(self, y_true, y_pred):
        """
        Prints per-class accuracy.
        Why: Standard report gives Precision/Recall, but sometimes we just want simple Accuracy:
        'Of all true BUYs, how many did we find?' (Recall aka Sensitivity)
        """
        print("\n" + "=" * 80)
        print("PER-CLASS PERFORMANCE ANALYSIS")
        print("=" * 80)
        for idx, name in enumerate(['HOLD (0)', 'BUY (1)', 'SELL (2)']):
            # Create a mask for the specific class (e.g., where True Label is BUY)
            mask = y_true == idx
            if mask.sum() > 0:
                # Calculate accuracy for that subset
                acc = (y_pred[mask] == idx).mean()
                print(f"{name}: {acc:.2%} accuracy ({mask.sum()} samples)")
