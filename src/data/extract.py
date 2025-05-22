from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.io_utils import save_if_needed


class ExtractYelpData(BaseEstimator, TransformerMixin):
    """
    Transformer for extracting and merging Yelp dataset components (users, reviews, businesses).
    Optionally saves the merged DataFrame to a parquet file.
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
        output_path_train: str | Path | None = None,
        output_path_infer: str | Path | None = None,
        chunksize: int = int(1e5),
        sample_size: int = int(1e6),
        train_ratio: float = (0.8),
    ) -> None:
        """
        Initialize ExtractYelpData.

        Args:
            user_path (Union[str, Path]): Path to the user JSON file.
            review_path (Union[str, Path]): Path to the review JSON file.
            business_path (Union[str, Path]): Path to the business JSON file.
            field_review (str): Name of the review count field in the user data.
            text (str): Name of the review text field in the review data.
            useful (str): Name of the usefulness score field in the review data.
            date (str): Name of the date field in the review data.
            user_id (str): Name of the user ID field in the user data.
            business_id (str): Name of the business ID field in the business data.
            output_path_train (Optional[Union[str, Path]], optional): Path to save the train dataset. Defaults to None.
            output_path_infer (Optional[Union[str, Path]], optional): Path to save the inference dataset. Defaults to None.
            chunksize (int, optional): Number of rows per chunk when reading JSON files. Defaults to 100_000.
            sample_size (int, optional): Number of most recent reviews to sample. Defaults to 1_000_100.
            train_ratio (float, optional): Ratio of training data to total data when splitting. Defaults to 0.8.
        """
        self.user_path: Path = Path(user_path)
        self.review_path: Path = Path(review_path)
        self.business_path: Path = Path(business_path)
        self.output_path_train: Path | None = (
            Path(output_path_train) if output_path_train else None
        )
        self.output_path_infer: Path | None = (
            Path(output_path_infer) if output_path_infer else None
        )
        self.chunksize: int = chunksize
        self.sample_size: int = sample_size
        self.field_review: str = field_review
        self.text: str = text
        self.useful: str = useful
        self.date: str = date
        self.user_id: str = user_id
        self.business_id: str = business_id
        self.train_ratio: float = train_ratio

    def fit(self, X: Any | None = None) -> "ExtractYelpData":
        """
        Fit method for compatibility with scikit-learn pipelines. Does nothing.

        Args:
            X: Ignored.

        Returns:
            ExtractYelpData: self
        """
        return self

    # Load user data
    def _load_user(self) -> pd.DataFrame:
        """
        Load user data, filtering users with at least one review.

        Returns:
            pd.DataFrame: Filtered user DataFrame.
        """
        logger.info(f"Loading user data from {self.user_path}")
        return pd.concat(
            [
                chunk[chunk[self.field_review] > 0]
                for chunk in pd.read_json(
                    self.user_path, lines=True, chunksize=self.chunksize
                )
            ],
            ignore_index=True,
        )

    # Load review data
    def _load_review(self) -> pd.DataFrame:
        """
        Load review data, filtering out empty reviews and those with zero usefulness.

        Returns:
            pd.DataFrame: Filtered review DataFrame.
        """
        logger.info(f"Loading review data from {self.review_path}")
        return pd.concat(
            [
                chunk[(chunk[self.text].str.strip() != "") & (chunk[self.useful] > 0)]
                for chunk in pd.read_json(
                    self.review_path, lines=True, chunksize=self.chunksize
                )
            ],
            ignore_index=True,
        )

    # Load business data
    def _load_business(self) -> pd.DataFrame:
        """
        Load business data.

        Returns:
            pd.DataFrame: Business DataFrame.
        """
        logger.info(f"Loading business data from {self.business_path}")
        return pd.read_json(self.business_path, lines=True)

    # Merge all data (user, review, business)
    def _merge_all(
        self,
        df_review: pd.DataFrame,
        df_user: pd.DataFrame,
        df_business: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge review, user, and business DataFrames.

        Args:
            df_review (pd.DataFrame): Review DataFrame.
            df_user (pd.DataFrame): User DataFrame.
            df_business (pd.DataFrame): Business DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        logger.info("Merging review, user, and business data")
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

    def transform(self, X: Any | None = None) -> pd.DataFrame:
        # 1. Load data
        logger.info("Loading data")
        df_user = self._load_user()
        df_review = self._load_review()
        df_business = self._load_business()

        # 2. Merge all data
        logger.info("Merging data")
        df_final = self._merge_all(df_review, df_user, df_business)

        # 3. Optional split into train/inference by count
        logger.info("Splitting data into train and inference sets")
        total_rows = len(df_final)
        train_rows = int(total_rows * self.train_ratio)

        df_train = df_final.iloc[:train_rows]
        df_infer = df_final.iloc[train_rows:]

        # 4. Save full dataset if output_path is given
        logger.info("Saving data")
        save_if_needed(df_train, self.output_path_train)
        save_if_needed(df_infer, self.output_path_infer)

        logger.success("Data extraction and merging completed successfully.")
        return df_train

    def set_output(self, *, transform: Any | None = None) -> "ExtractYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.

        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            ExtractYelpData: self
        """
        return self
