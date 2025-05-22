from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.utils import estimator_html_repr

from src.model.training import TrainModelTransformer
from src.model.validation import evaluate_model
from src.model.mdt import MDTYelpData
from src.model.mdt import split_data


BASE_DIR = Path(__file__).resolve().parents[3]
config = OmegaConf.load(BASE_DIR / "conf/model_training/training.yml")


def run_training_pipeline(data: pd.DataFrame) -> Any:
    """
    Run the full training pipeline: split, transform, train, validate and return model.

    Args:
        data: DataFrame with the full dataset (features + target).

    Returns:
        Trained model if validation passes.
    """
    # Split
    X_train, X_test, y_train, y_test = split_data(
        df=data,
        target_column=config.target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # MDT
    mdt = MDTYelpData(
        corr_threshold=config.reducer.corr_threshold,
        importance_threshold=config.reducer.importance_threshold,
        scoring=config.reducer.scoring,
        random_state=config.random_state,
        target_column=config.target_column,
    )

    # Training
    model_trainer = TrainModelTransformer(
        classifier_fn=XGBClassifier,
        best_params=dict(config.model_params),
    )

    pipeline = Pipeline([
        ("mdt", mdt),
        ("train", model_trainer),
    ])

    # Fit
    pipeline.fit(X_train, y_train)

    # Save html representation of the pipeline
    pipeline_repr_path = BASE_DIR / config.mit_output_transform_path
    with open(pipeline_repr_path, "w") as f:
        f.write(estimator_html_repr(pipeline))


    # Validate
    results = evaluate_model(
        model=model_trainer.model_,
        X_test=mdt.transform(X_test),
        y_test=y_test,
        thresholds=dict(config.metric_thresholds),
    )

    # Save model and metrics
    joblib.dump(model_trainer.model_, BASE_DIR / config.output_model_path)
    pd.DataFrame([results]).to_parquet(BASE_DIR / config.output_metrics_path, index=False)

    return model_trainer.model_
