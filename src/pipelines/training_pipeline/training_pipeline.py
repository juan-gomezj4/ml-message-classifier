from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from xgboost import XGBClassifier

from src.model.mdt import MDTYelpData, split_data, transform_stars_to_target
from src.model.training import TrainModelTransformer
from src.model.validation import evaluate_and_save_model
from src.utils.io_utils import save_pipeline_if_needed

# Base directory for relative paths
BASE_DIR = Path(__file__).resolve().parents[3]

# Load configuration files
training_config = OmegaConf.load(BASE_DIR / "conf/model_training/training.yml")


# Build pipeline
def run_training_pipeline() -> None:
    """
    Run the training pipeline and return the final DataFrame.

    Returns:
        None
    """
    logger.info("Running training pipeline...")

    # Step 1: Load feature data
    logger.info("Loading feature data for training...")
    feature_data = pd.read_parquet(
        BASE_DIR / training_config.training.feature_data_path
    )

    # Step 2: Create target
    df_target = transform_stars_to_target(
        feature_data, training_config.mdt.stars_column
    )

    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(
        df_target,
        target_column=training_config.mdt.target_column,
        test_size=training_config.training.test_size,
        random_state=training_config.training.random_state,
        output_path_train=training_config.output_path_train,
        output_path_test=training_config.output_path_test,
        output_path_ytrain=training_config.output_path_ytrain,
        output_path_ytest=training_config.output_path_ytest,
    )

    # Step 4: Load classifier and parameters
    logger.info("Loading classifier and parameters...")
    classifier_registry = {
        "XGBClassifier": XGBClassifier,
    }
    classifier_fn = TrainModelTransformer.load_model_name(
        model_name_path=Path(training_config.training.classifier_fn_path),
        classifier_registry=classifier_registry,
    )
    best_params = TrainModelTransformer.load_model_parameters(
        params_path=training_config.training.best_params_path
    )

    # Step 5: Build pipeline
    logger.info("Building pipeline...")
    pipeline = Pipeline(
        steps=[
            (
                "mdt",
                MDTYelpData(
                    corr_threshold=training_config.mdt.corr_threshold,
                    importance_threshold=training_config.mdt.importance_threshold,
                    scoring=training_config.mdt.scoring,
                    random_state=training_config.mdt.random_state,
                    target_column=training_config.mdt.target_column,
                    output_path_fit=training_config.mdt_output_path_fit,
                    output_path_transformed=training_config.mdt_output_path_transform,
                ),
            ),
            (
                "training",
                TrainModelTransformer(
                    classifier_fn=classifier_fn,
                    best_params=best_params,
                ),
            ),
        ]
    )

    # Step 6: Fit pipeline
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # Step 7: Transform test set
    logger.info("Transforming test set...")
    X_test_transformed = pipeline.transform(X_test)

    # Step 8: Save pipeline trained
    logger.info("Saving trained pipeline...")
    save_pipeline_if_needed(pipeline, training_config.training_output_pipeline_path)

    # Step 9: Save the pipeline view
    with open(BASE_DIR / training_config.training_output_transform_path, "w") as f:
        f.write(estimator_html_repr(pipeline))

    # Step 10: Validator
    try:
        evaluate_and_save_model(
            model=pipeline.named_steps["training"].model_,
            X_test=X_test_transformed,
            y_test=y_test,
            thresholds={
                "f1": training_config.validate.f1_macro_threshold,
                "recall": training_config.validate.recall_threshold,
            },
            output_path=BASE_DIR / training_config.validate_output_final_model_path,
        )
    except ValueError as e:
        logger.warning(f"Model validation failed: {e}")

    logger.info("Training pipeline completed.")
