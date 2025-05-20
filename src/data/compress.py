from typing import Any, List, Optional

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drop columns from a DataFrame.
    """

    def __init__(self, columns: List[str]) -> None:
        """
        Args:
            columns: List of column names to drop.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame) -> "DropColumnsTransformer":
        """
        No fitting is needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            DropColumnsTransformer: Returns self.
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
        return X.drop(columns=self.columns, errors="ignore")


class CompressYelpData(BaseEstimator, TransformerMixin):
    """
    Compress the DataFrame by dropping specified columns and saving the result if configured.
    """

    def __init__(
        self,
        categorical: List[str],
        numerical: List[str],
        string: List[str],
        data: List[str],
        output_path: Optional[str] = None,
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
        self.data = data
        self.output_path = output_path

        # Create transformers for each group of columns to drop
        self.group_droppers: List[DropColumnsTransformer] = [
            DropColumnsTransformer(columns=self.categorical),
            DropColumnsTransformer(columns=self.numerical),
            DropColumnsTransformer(columns=self.string),
            DropColumnsTransformer(columns=self.data),
        ]

    def fit(self, X: pd.DataFrame) -> "CompressYelpData":
        """
        No fitting is needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            CompressYelpData: Returns self.
        """
        return self

    # Save data to parquet if output path is specified
    def _save_if_needed(self, df: pd.DataFrame) -> None:
        """
        Save the DataFrame to parquet if output_path is set.

        Args:
            df (pd.DataFrame): DataFrame to save.
        """
        if self.output_path:
            logger.info(f"Saving merged data to {self.output_path}")
            df.to_parquet(self.output_path, index=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all drop transformers to the DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Compressed DataFrame with specified columns dropped.
        """
        # Apply all drop transformers
        logger.info("Compressing DataFrame by dropping specified columns.")
        X_compressed = X.copy()
        for dropper in self.group_droppers:
            X_compressed = dropper.transform(X_compressed)
        logger.info("DataFrame compressed successfully.")

        # Save if needed
        logger.info("Saving compressed DataFrame if output path is specified.")
        self._save_if_needed(X_compressed)

        return X_compressed

    def set_output(self, *, transform: Optional[Any] = None) -> "CompressYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.

        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            CompressYelpData: self
        """
        return self
