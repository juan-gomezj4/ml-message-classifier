from typing import Any

import pandas as pd
from sklearn.metrics import classification_report


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """
    Evaluate a classification model and verify that performance meets specified thresholds.

    Args:
        model: Trained model with `.predict()` method.
        X_test: Features for validation.
        y_test: Ground truth labels.
        thresholds: Dict of metric thresholds, e.g., {'f1_macro': 0.75}.

    Returns:
        Dictionary with metric results.

    Raises:
        ValueError: If any metric is below its threshold.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    macro_avg = report.get("macro avg", {})
    results: dict[str, float] = {}

    for metric, threshold in thresholds.items():
        metric_key = metric.replace("_macro", "")
        score = macro_avg.get(metric_key)

        if score is None:
            raise ValueError(f"Metric '{metric}' not found in evaluation results.")

        results[metric] = score

        if score < threshold:
            raise ValueError(
                f"[FAIL] {metric} = {score:.4f} < threshold = {threshold:.4f}"
            )

    return results
