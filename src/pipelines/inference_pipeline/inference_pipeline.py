from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.data.validate import ValidateYelpData
from src.data.aggregate import AggregateYelpData
from src.data.compress import CompressYelpData
from src.data.mit import MITYelpData


BASE_DIR = Path(__file__).resolve().parents[3]
config_feature = OmegaConf.load(BASE_DIR / "conf/data_feature/feature.yml")
config_inference = OmegaConf.load(BASE_DIR / "conf/model_inference/inference.yml")


def run_inference_pipeline(data: pd.DataFrame, model: Any) -> None:
    """
    Run the full inference pipeline: load data and model, MIT, MDT, predict.

    Args:
        data: Input data as a DataFrame
        model: Trained model
    """
    logger.info("Starting inference pipeline")

    # MIT - Apply model-independent transformations
    # Validate
    logger.info("Validating inference data")
    validate = ValidateYelpData(
        drop_columns=config_feature.validation_drop_columns,
        drop_columns_na=config_feature.validation.cols_drop_na,
        cols_categoric=config_feature.validation.cols_categoric,
        cols_numeric_float=config_feature.validation.cols_numeric_float,
        cols_numeric_int=config_feature.validation.cols_numeric_int,
        cols_boolean=config_feature.validation.cols_boolean,
        cols_string=config_feature.validation.cols_string,
        col_date=config_feature.validation.col_date,
    )
    df = validate.transform(data)
    logger.info("Data validation complete")

    # Aggregate
    logger.info("Applying aggregation transformations")
    aggregate = AggregateYelpData(
        elite=config_feature.aggregate_categorical.elite,
        elite_count=config_feature.aggregate_categorical.elite_count,
        frequency_encode=config_feature.aggregate_categorical.frequency_encode,
        binary_flag=config_feature.aggregate_numerical.binary_flag,
        qcut_level=config_feature.aggregate_numerical.qcut_level,
        fans=config_feature.aggregate_numerical.fans,
        text=config_feature.aggregate_string.text,
        text_length=config_feature.aggregate_string.text_length,
        categories=config_feature.aggregate_string.categories,
        date=config_feature.aggregate_date.date,
    )
    df = aggregate.transform(df)
    logger.info("Aggregation complete")

    # Save review id
    logger.info("Saving review IDs")
    review_id_col = config_inference.review_id
    review_ids = df[review_id_col].copy()

    # Compress
    logger.info("Applying compression transformations")
    compress = CompressYelpData(
        categorical=config_feature.compress_categorical,
        numerical=config_feature.compress_numerical,
        string=config_feature.compress_string,
        date=config_feature.compress_date,
    )
    df = compress.transform(df)
    logger.info("Compression complete")

    # MIT
    logger.info("Applying model-independent transformations")
    mit = MITYelpData(
        text_column=config_feature.mit.text_column,
        group_col=config_feature.mit.group_col,
        value_col=config_feature.mit.value_col,
        embedding_model=config_feature.mit.embedding_model,
        n_components=config_feature.mit.n_components,
    )
    df = mit.transform(df)
    logger.info("MIT transformations complete")

    # MDT - Transform with fitted MDT
    logger.info("Applying model-dependent transformations")
    df = pipeline.named_steps["mdt"].transform(df)
    logger.info("MDT transformations complete")

    # Predict
    logger.info("Making predictions")
    y_pred = pipeline.named_steps["train"].model_.predict(df)
    y_proba = pipeline.named_steps["train"].model_.predict_proba(df)
    logger.info("Predictions complete")

    # Build dataframe with predictions
    logger.info("Building output dataframe")
    class_labels = pipeline.named_steps["train"].model_.classes_
    proba_df = pd.DataFrame(y_proba, columns=[f"proba_{c}" for c in class_labels])
    output_df = pd.DataFrame({
        review_id_col: review_ids,
        "prediction": y_pred
    }).join(proba_df)

    # Save dataframe with predictions
    logger.info(f"Saving predictions to {config_inference.paths.prediction}")
    output_df.to_parquet(config_inference.paths.prediction, index=False)
    logger.success(f"Inference pipeline complete. Processed {len(output_df)} records")
