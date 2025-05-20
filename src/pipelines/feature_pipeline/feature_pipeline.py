from pathlib import Path

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from src.data.aggregate import AggregateYelpData
from src.data.compress import CompressYelpData
from src.data.extract import ExtractYelpData
from src.data.mit import MITYelpData
from src.data.validate import ValidateYelpData

BASE_DIR = Path(__file__).resolve().parents[3]

# Load configurations
extract_config = OmegaConf.load(BASE_DIR / "conf/data_feature/extract.yml")
validate_config = OmegaConf.load(BASE_DIR / "conf/data_feature/validate.yml")
aggregate_config = OmegaConf.load(BASE_DIR / "conf/data_feature/aggregate.yml")
compress_config = OmegaConf.load(BASE_DIR / "conf/data_feature/compress.yml")
mit_config = OmegaConf.load(BASE_DIR / "conf/data_feature/mit.yml")

# Build pipeline
feature_pipeline = Pipeline(
    [
        (
            "extract",
            ExtractYelpData(
                user_path=BASE_DIR / extract_config.paths.user,
                review_path=BASE_DIR / extract_config.paths.review,
                business_path=BASE_DIR / extract_config.paths.business,
                output_path=BASE_DIR / extract_config.paths.output,
                chunksize=extract_config.params.chunksize,
                sample_size=extract_config.params.sample_size,
                field_review=extract_config.fields.review_count,
                text=extract_config.fields.text,
                useful=extract_config.fields.useful,
                date=extract_config.fields.date,
                user_id=extract_config.fields.user_id,
                business_id=extract_config.fields.business_id,
            ),
        ),
        (
            "validate",
            ValidateYelpData(
                drop_columns=validate_config.drop_columns,
                drop_columns_na=validate_config.cols_drop_na,
                cols_categoric=validate_config.cols_categoric,
                cols_numeric_float=validate_config.cols_numeric_float,
                cols_numeric_int=validate_config.cols_numeric_int,
                cols_boolean=validate_config.cols_boolean,
                cols_string=validate_config.cols_string,
                col_date=validate_config.col_date,
                output_path=BASE_DIR / validate_config.output_path,
            ),
        ),
        (
            "aggregate",
            AggregateYelpData(
                elite=aggregate_config.categorical.elite,
                elite_count=aggregate_config.categorical.elite_count,
                frequency_encode=aggregate_config.categorical.frequency_encode,
                binary_flag=aggregate_config.numerical.binary_flag,
                qcut_level=aggregate_config.numerical.qcut_level,
                fans=aggregate_config.numerical.fans,
                text=aggregate_config.string.text,
                text_length=aggregate_config.string.text_length,
                categories=aggregate_config.string.categories,
                date=aggregate_config.date.date,
                output_path=BASE_DIR / aggregate_config.output_path,
            ),
        ),
        (
            "compress",
            CompressYelpData(
                categorical=compress_config.categorical,
                numerical=compress_config.numerical,
                string=compress_config.string,
                data=compress_config.data,
                output_path=BASE_DIR / compress_config.output_path,
            ),
        ),
        (
            "mit",
            MITYelpData(
                text_column=mit_config.text_column,
                group_col=mit_config.group_col,
                value_col=mit_config.value_col,
                embedding_model=mit_config.embedding_model,
                n_components=mit_config.n_components,
                output_path=BASE_DIR / mit_config.output_path,
            ),
        ),
    ]
)
