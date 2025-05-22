from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractYelpData(BaseEstimator, TransformerMixin):
    """
    Extracts and merges Yelp dataset components (users, reviews, businesses).
    Returns a single merged DataFrame for downstream processing.
    """

    def __init__(
        self,
        user_path: str | Path,
        review_path: str | Path,
        business_path: str | Path,
        field_review: str,
        text: str,
        useful: str,
        date: str,
        user_id: str,
        business_id: str,
        chunksize: int = int(1e5),
        sample_size: int = int(1e6),
    ) -> None:
        self.user_path = Path(user_path)
        self.review_path = Path(review_path)
        self.business_path = Path(business_path)
        self.field_review = field_review
        self.text = text
        self.useful = useful
        self.date = date
        self.user_id = user_id
        self.business_id = business_id
        self.chunksize = chunksize
        self.sample_size = sample_size

    def fit(self, X: Any = None, y: Any = None) -> "ExtractYelpData":
        return self

    def transform(self, X: Any = None) -> pd.DataFrame:
        logger.info("Extracting Yelp data...")

        df_user = self._load_user()
        df_review = self._load_review()
        df_business = self._load_business()

        df_merged = self._merge_all(df_review, df_user, df_business)

        logger.success("Extraction completed successfully.")
        return df_merged

    def _load_user(self) -> pd.DataFrame:
        logger.info(f"Loading users from {self.user_path}")
        return pd.concat(
            [
                chunk[chunk[self.field_review] > 0]
                for chunk in pd.read_json(
                    self.user_path, lines=True, chunksize=self.chunksize
                )
            ],
            ignore_index=True,
        )

    def _load_review(self) -> pd.DataFrame:
        logger.info(f"Loading reviews from {self.review_path}")
        return pd.concat(
            [
                chunk[(chunk[self.text].str.strip() != "") & (chunk[self.useful] > 0)]
                for chunk in pd.read_json(
                    self.review_path, lines=True, chunksize=self.chunksize
                )
            ],
            ignore_index=True,
        )

    def _load_business(self) -> pd.DataFrame:
        logger.info(f"Loading businesses from {self.business_path}")
        return pd.read_json(self.business_path, lines=True)

    def _merge_all(
        self, df_review: pd.DataFrame, df_user: pd.DataFrame, df_business: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Merging data...")
        df_review = df_review.sort_values(self.date, ascending=True).head(
            self.sample_size
        )
        df_merged = df_review.merge(
            df_user, on=self.user_id, how="left", suffixes=("", "_user")
        )
        df_merged = df_merged.merge(
            df_business, on=self.business_id, how="left", suffixes=("", "_business")
        )
        return df_merged
