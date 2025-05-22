from typing import Any

import joblib
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report


def evaluate_and_save_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: dict[str, float],
    output_path: str,
) -> None:
    """
    Evaluate a trained model against threshold metrics and save it if passed.

    Args:
        model (Any): Trained model with `.predict` method.
        X_test (pd.DataFrame): Features for validation.
        y_test (pd.Series): Ground truth labels.
        thresholds (Dict[str, float]): Minimum thresholds for metrics like 'f1_macro'.
        output_path (str): Path to save the model if thresholds are met.

    Raises:
        ValueError: If any metric is below its threshold.
    """
    logger.info("Validator model started")

    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)

    logger.info("Reporting metrics...")
    report = classification_report(y_test, y_pred, output_dict=True)

    macro_avg = report.get("macro avg", {})
    for metric, threshold in thresholds.items():
        metric_name = metric.replace("_macro", "")
        score = macro_avg.get(metric_name)

        logger.info(f"Evaluating {metric} with threshold {threshold}...")
        if score is None:
            raise ValueError(f"[ERROR] Metric {metric} not found in report.")
        if score < threshold:
            raise ValueError(
                f"[FAIL] {metric} = {score:.4f} < threshold = {threshold:.4f}"
            )
        logger.info(f"[PASS] {metric} = {score:.4f} ≥ threshold = {threshold:.4f}")

    joblib.dump(model, output_path)
    logger.success(f"Model saved to: {output_path}")
