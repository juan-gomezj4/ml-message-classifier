from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from xgboost import XGBClassifier

from src.model.mdt import MDTYelpData, split_data, transform_stars_to_target
from src.model.training import TrainModelTransformer
from src.model.validation import evaluate_model

BASE_DIR = Path(__file__).resolve().parents[3]
config_feature = OmegaConf.load(BASE_DIR / "conf/data_feature/feature.yml")
config_training = OmegaConf.load(BASE_DIR / "conf/model_training/training.yml")


def run_training_pipeline() -> Any:
    """
    Run the full training pipeline: split, transform, train, validate and return model.

    Args:
        data: DataFrame with the full dataset (features + target).

    Returns:
        Trained model if validation passes.
    """
    # Load data feature
    data = pd.read_parquet(config_feature.mit_output_path2)

    # Create target column
    data = transform_stars_to_target(data, config_training.target_column)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df=data,
        target_column=config_training.target_column,
        test_size=config_training.test_size,
        random_state=config_training.random_state,
    )

    # MDT
    mdt = MDTYelpData(
        corr_threshold=config_training.mdt.corr_threshold,
        importance_threshold=config_training.mdt.importance_threshold,
        scoring=config_training.mdt.scoring,
        random_state=config_training.random_state,
        target_column=config_training.target_column,
    )

    # Training
    model_trainer = TrainModelTransformer(
        classifier_fn=XGBClassifier,
        best_params=dict(config_training.model.params),
    )

    pipeline = Pipeline(
        [
            ("mdt", mdt),
            ("train", model_trainer),
        ]
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Save html representation of the pipeline
    pipeline_repr_path = (
        BASE_DIR / config_training.outputs.training_output_transform_path
    )
    with open(pipeline_repr_path, "w") as f:
        f.write(estimator_html_repr(pipeline))

    # Extract and transformed test set
    mdt_fitted = pipeline.named_steps["mdt"]
    X_test_transformed = mdt_fitted.transform(X_test)

    # Extract the trained model
    trained_model = pipeline.named_steps["train"].model_

    # Validate
    results = evaluate_model(
        model=trained_model,
        X_test=X_test_transformed,
        y_test=y_test,
        thresholds=dict(config_training.metric_thresholds),
    )

    # Save model and metrics
    joblib.dump(pipeline, BASE_DIR / config_training.outputs.model_path)
    pd.DataFrame([results]).to_parquet(
        BASE_DIR / config_training.outputs.metrics_path, index=False
    )

    return model_trainer.model_
