from typing import Any

import pandas as pd
from sklearn.metrics import classification_report

METRIC_MAP = {
    "f1_macro": "f1-score",
    "recall_macro": "recall",
}


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
        Dictionary with computed metric values.

    Raises:
        ValueError: If a metric is missing or below its threshold.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_avg = report.get("macro avg", {})

    results: dict[str, float] = {}

    for metric, threshold in thresholds.items():
        key_in_macro = METRIC_MAP.get(metric)
        if key_in_macro is None:
            msg = f"Unsupported metric '{metric}'. Please update METRIC_MAP if needed."
            raise ValueError(msg)

        score = macro_avg.get(key_in_macro)
        if score is None:
            msg = f"Metric '{metric}' not found in evaluation results."
            raise ValueError(msg)

        results[metric] = score

        if score < threshold:
            msg = f"[FAIL] {metric} = {score:.4f} < threshold = {threshold:.4f}"
            raise ValueError(msg)

    return results
