from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class CompressYelpData(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from a DataFrame to reduce size.
    """

    def __init__(
        self,
        categorical: list[str],
        numerical: list[str],
        string: list[str],
        date: list[str],
    ) -> None:
        """
        Args:
            categorical: Categorical columns to drop.
            numerical: Numerical columns to drop.
            string: String columns to drop.
            date: Date columns to drop.
        """
        self.categorical = categorical
        self.numerical = numerical
        self.string = string
        self.date = date

    def fit(self, X: Any = None, y: Any = None) -> "CompressYelpData":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Dropping columns: {self.categorical + self.numerical + self.string + self.date}")
        # Step 1: Drop specified columns
        X_compressed = X.drop(columns=self.categorical + self.numerical + self.string + self.date, errors="ignore")
        logger.success("DataFrame compressed.")
        return X_compressed
