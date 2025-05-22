from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.io_utils import save_if_needed


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drop columns from a DataFrame.
    """

    def __init__(self, columns: list[str]) -> None:
        """
        Args:
            columns: List of column names to drop.
        """
        self.columns = columns

    def fit(self, X: Any | None = None) -> "DropColumnsTransformer":
        """
        Fit method for compatibility with scikit-learn pipelines. Does nothing.

        Args:
            X: Ignored.

        Returns:
            DropColumnsTransformer: self
        """
        return self

    # Drop columns from the DataFrame
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from the input DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        logger.info(f"Dropping columns: {self.columns}")
        return X.drop(columns=self.columns, errors="ignore")


class CompressYelpData(BaseEstimator, TransformerMixin):
    """
    Compress the DataFrame by dropping specified columns and saving the result if configured.
    """

    def __init__(
        self,
        categorical: list[str],
        numerical: list[str],
        string: list[str],
        date: list[str],
        output_path: str | None = None,
    ) -> None:
        """
        Initialize the CompressYelpData transformer.
        Args:
            categorical (List[str]): List of categorical columns to drop.
            numerical (List[str]): List of numerical columns to drop.
            string (List[str]): List of string columns to drop.
            data (List[str]): List of data columns to drop.
            output_path (Optional[str]): Path to save the compressed DataFrame. If None, do not save.
        """
        self.categorical = categorical
        self.numerical = numerical
        self.string = string
        self.date = date
        self.output_path = output_path

        # Create transformers for each group of columns to drop
        self.group_droppers: list[DropColumnsTransformer] = [
            DropColumnsTransformer(columns=self.categorical),
            DropColumnsTransformer(columns=self.numerical),
            DropColumnsTransformer(columns=self.string),
            DropColumnsTransformer(columns=self.date),
        ]

    def fit(self, X: Any | None = None) -> "CompressYelpData":
        """
        Fit method for compatibility with scikit-learn pipelines. Does nothing.

        Args:
            X: Ignored.

        Returns:
            CompressYelpData: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all drop transformers to the DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Compressed DataFrame with specified columns dropped.
        """
        # Step 1: Apply all drop transformers
        logger.info("Compressing DataFrame by dropping specified columns.")
        X_compressed = X.copy()
        for dropper in self.group_droppers:
            X_compressed = dropper.transform(X_compressed)
        logger.info(
            f"Columns dropped: {self.categorical + self.numerical + self.string + self.date}"
        )
        logger.info("DataFrame compressed successfully.")

        # Step 2: Save if needed
        logger.info("Saving compressed DataFrame if output path is specified.")
        save_if_needed(X_compressed, self.output_path)

        return X_compressed

    def set_output(self, *, transform: Any | None = None) -> "CompressYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.

        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            CompressYelpData: self
        """
        return self
