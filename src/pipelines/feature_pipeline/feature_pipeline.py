from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

from src.data.aggregate import AggregateYelpData
from src.data.compress import CompressYelpData
from src.data.extract import ExtractYelpData
from src.data.mit import MITYelpData
from src.data.validate import ValidateYelpData

BASE_DIR = Path(__file__).resolve().parents[3]

# Load configurations
feature_config = OmegaConf.load(BASE_DIR / "conf/data_feature/feature.yml")


def run_feature_pipeline() -> pd.DataFrame:
    """
    Build and run the feature pipeline and return the final DataFrame.

    Returns:
        pd.DataFrame: Fully processed features after all FTI steps.
    """
    logger.info("Running feature pipeline...")

    # Step 1: Process data
    # Step 2: Validate data
    # Step 3: Aggregate data
    # Step 4: Compress data
    # Step 5: Apply MITYelpData
    logger.info("Building feature pipeline...")
    pipeline = Pipeline(
        steps=[
            (
                "extract",
                ExtractYelpData(
                    user_path=BASE_DIR / feature_config.paths.user,
                    review_path=BASE_DIR / feature_config.paths.review,
                    business_path=BASE_DIR / feature_config.paths.business,
                    output_path_train=BASE_DIR
                    / feature_config.extract_output_path_train,
                    output_path_infer=BASE_DIR
                    / feature_config.extract_output_path_infer,
                    chunksize=feature_config.data_params.chunksize,
                    sample_size=feature_config.data_params.sample_size,
                    field_review=feature_config.data_columns.review_count,
                    text=feature_config.data_columns.text,
                    useful=feature_config.data_columns.useful,
                    date=feature_config.data_columns.date,
                    user_id=feature_config.data_columns.user_id,
                    business_id=feature_config.data_columns.business_id,
                    train_ratio=feature_config.data_params.train_ratio,
                ),
            ),
            (
                "validate",
                ValidateYelpData(
                    drop_columns=feature_config.validation_drop_columns,
                    drop_columns_na=feature_config.validation.cols_drop_na,
                    cols_categoric=feature_config.validation.cols_categoric,
                    cols_numeric_float=feature_config.validation.cols_numeric_float,
                    cols_numeric_int=feature_config.validation.cols_numeric_int,
                    cols_boolean=feature_config.validation.cols_boolean,
                    cols_string=feature_config.validation.cols_string,
                    col_date=feature_config.validation.col_date,
                    output_path=BASE_DIR / feature_config.validate_output_path,
                ),
            ),
            (
                "aggregate",
                AggregateYelpData(
                    elite=feature_config.aggregate_categorical.elite,
                    elite_count=feature_config.aggregate_categorical.elite_count,
                    frequency_encode=feature_config.aggregate_categorical.frequency_encode,
                    binary_flag=feature_config.aggregate_numerical.binary_flag,
                    qcut_level=feature_config.aggregate_numerical.qcut_level,
                    fans=feature_config.aggregate_numerical.fans,
                    text=feature_config.aggregate_string.text,
                    text_length=feature_config.aggregate_string.text_length,
                    categories=feature_config.aggregate_string.categories,
                    date=feature_config.aggregate_date.date,
                    output_path=BASE_DIR / feature_config.aggregate_output_path,
                ),
            ),
            (
                "compress",
                CompressYelpData(
                    categorical=feature_config.compress_categorical,
                    numerical=feature_config.compress_numerical,
                    string=feature_config.compress_string,
                    date=feature_config.compress_date,
                    output_path=BASE_DIR / feature_config.compress_output_path,
                ),
            ),
            (
                "mit",
                MITYelpData(
                    text_column=feature_config.mit.text_column,
                    group_col=feature_config.mit.group_col,
                    value_col=feature_config.mit.value_col,
                    embedding_model=feature_config.mit.embedding_model,
                    n_components=feature_config.mit.n_components,
                    output_path=BASE_DIR / feature_config.mit_output_path,
                ),
            ),
        ]
    )

    # Step 6: Fit and transform the pipeline
    logger.info("Fitting and transforming the pipeline...")
    dataframe: pd.DataFrame = pipeline.fit_transform(None)

    # Step 7: Save the pipeline view
    logger.info("Saving the pipeline view...")
    with open(BASE_DIR / feature_config.mit_output_transform_path, "w") as f:
        f.write(estimator_html_repr(pipeline))

    logger.info("Feature pipeline completed.")
    return dataframe
