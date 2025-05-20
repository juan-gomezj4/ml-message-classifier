from pathlib import Path

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from src.data.extract import ExtractYelpData

BASE_DIR = Path(__file__).resolve().parents[3]
config = OmegaConf.load(BASE_DIR / "conf/data_extraction/extract.yml")

feature_pipeline = Pipeline(
    [
        (
            "extract",
            ExtractYelpData(
                user_path=BASE_DIR / config.paths.user,
                review_path=BASE_DIR / config.paths.review,
                business_path=BASE_DIR / config.paths.business,
                output_path=BASE_DIR / config.paths.output,
                chunksize=config.params.chunksize,
                sample_size=config.params.sample_size,
                field_review=config.fields.review_count,
                text=config.fields.text,
                useful=config.fields.useful,
                date=config.fields.date,
                user_id=config.fields.user_id,
                business_id=config.fields.business_id,
            ),
        ),
    ]
)
