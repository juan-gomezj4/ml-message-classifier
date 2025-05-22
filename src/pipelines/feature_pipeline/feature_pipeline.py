from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

from src.data.extract import ExtractYelpData
from src.data.validate import ValidateYelpData
from src.data.aggregate import AggregateYelpData
from src.data.compress import CompressYelpData
from src.data.mit import MITYelpData

BASE_DIR = Path(__file__).resolve().parents[3]
config = OmegaConf.load(BASE_DIR / "conf/data_feature/feature.yml")


def run_feature_pipeline() -> pd.DataFrame:
    logger.info("Running feature pipeline...")

    # 1. Extract
    extract = ExtractYelpData(
        user_path=BASE_DIR / config.paths.user,
        review_path=BASE_DIR / config.paths.review,
        business_path=BASE_DIR / config.paths.business,
        chunksize=config.data_params.chunksize,
        sample_size=config.data_params.sample_size,
        field_review=config.data_columns.review_count,
        text=config.data_columns.text,
        useful=config.data_columns.useful,
        date=config.data_columns.date,
        user_id=config.data_columns.user_id,
        business_id=config.data_columns.business_id,
    )
    df = extract.fit_transform(None)
    df.to_parquet(BASE_DIR / config.extract_output_path_train, index=False)

    # 2. Validate
    validate = ValidateYelpData(
        drop_columns=config.validation_drop_columns,
        drop_columns_na=config.validation.cols_drop_na,
        cols_categoric=config.validation.cols_categoric,
        cols_numeric_float=config.validation.cols_numeric_float,
        cols_numeric_int=config.validation.cols_numeric_int,
        cols_boolean=config.validation.cols_boolean,
        cols_string=config.validation.cols_string,
        col_date=config.validation.col_date,
    )
    df = validate.transform(df)
    df.to_parquet(BASE_DIR / config.validate_output_path, index=False)

    # 3. Aggregate
    aggregate = AggregateYelpData(
        elite=config.aggregate_categorical.elite,
        elite_count=config.aggregate_categorical.elite_count,
        frequency_encode=config.aggregate_categorical.frequency_encode,
        binary_flag=config.aggregate_numerical.binary_flag,
        qcut_level=config.aggregate_numerical.qcut_level,
        fans=config.aggregate_numerical.fans,
        text=config.aggregate_string.text,
        text_length=config.aggregate_string.text_length,
        categories=config.aggregate_string.categories,
        date=config.aggregate_date.date,
    )
    df = aggregate.transform(df)
    df.to_parquet(BASE_DIR / config.aggregate_output_path, index=False)

    # 4. Compress
    compress = CompressYelpData(
        categorical=config.compress_categorical,
        numerical=config.compress_numerical,
        string=config.compress_string,
        date=config.compress_date,
    )
    df = compress.transform(df)
    df.to_parquet(BASE_DIR / config.compress_output_path, index=False)

    # 5. MIT
    mit = MITYelpData(
        text_column=config.mit.text_column,
        group_col=config.mit.group_col,
        value_col=config.mit.value_col,
        embedding_model=config.mit.embedding_model,
        n_components=config.mit.n_components,
    )
    df = mit.transform(df)
    df.to_parquet(BASE_DIR / config.mit_output_path, index=False)

    # 6. Save HTML representation of the pipeline
    pipeline_repr_path = BASE_DIR / config.mit_output_transform_path
    pipeline = Pipeline(
        [
            ("extract", extract),
            ("validate", validate),
            ("aggregate", aggregate),
            ("compress", compress),
            ("mit", mit),
        ]
    )
    with open(pipeline_repr_path, "w") as f:
        f.write(estimator_html_repr(pipeline))
    logger.info(f"Saved pipeline view to {pipeline_repr_path}")

    logger.success("Feature pipeline completed.")
    return df
