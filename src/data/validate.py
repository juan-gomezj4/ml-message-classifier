from typing import Any

import pandas as pd
from loguru import logger
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin


class ValidateYelpData(BaseEstimator, TransformerMixin):
    """
    Transformer to clean and validate column types for Yelp dataset.
    """

    def __init__(
        self,
        drop_columns: list[str],
        drop_columns_na: list[str],
        cols_categoric: list[str],
        cols_numeric_float: list[str],
        cols_numeric_int: list[str],
        cols_boolean: list[str],
        cols_string: list[str],
        col_date: str,
    ) -> None:
        self.drop_columns = drop_columns
        self.drop_columns_na = drop_columns_na
        self.cols_categoric = cols_categoric
        self.cols_numeric_float = cols_numeric_float
        self.cols_numeric_int = cols_numeric_int
        self.cols_boolean = cols_boolean
        self.cols_string = cols_string
        self.col_date = col_date

    def fit(self, X: Any = None, y: Any = None) -> "ValidateYelpData":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating Yelp data...")

        # 1. Drop columns
        X = X.drop(columns=self.drop_columns, errors="ignore")
        X = X.dropna(subset=self.drop_columns_na)

        # 2. Convert types
        self._safe_cast(X, self.cols_categoric, "category")
        self._safe_cast(X, self.cols_numeric_float, "float")
        self._safe_cast(X, self.cols_numeric_int, "int32")
        self._safe_cast(X, self.cols_boolean, "bool")
        self._safe_cast(X, self.cols_string, "object")

        if self.col_date in X.columns:
            X[self.col_date] = pd.to_datetime(X[self.col_date], errors="coerce")
        else:
            logger.warning(f"Date column '{self.col_date}' not found.")

        # 3. Type checks
        self._verify_types(X)

        logger.success("Validation completed.")
        return X

    def _safe_cast(self, df: pd.DataFrame, cols: list[str], dtype: Any) -> None:
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            logger.debug(f"Casting to {dtype}: {valid_cols}")
            df[valid_cols] = df[valid_cols].astype(dtype)

    def _verify_types(self, X: pd.DataFrame) -> None:
        checks = {
            **{col: is_string_dtype(X[col]) for col in self.cols_categoric if col in X},
            **{
                col: is_float_dtype(X[col])
                for col in self.cols_numeric_float
                if col in X
            },
            **{
                col: is_integer_dtype(X[col])
                for col in self.cols_numeric_int
                if col in X
            },
            **{col: is_bool_dtype(X[col]) for col in self.cols_boolean if col in X},
            **{
                col: is_string_dtype(X[col]) or is_object_dtype(X[col])
                for col in self.cols_string
                if col in X
            },
        }

        if self.col_date in X.columns:
            checks[self.col_date] = is_datetime64_dtype(X[self.col_date])

        failed = [col for col, valid in checks.items() if not valid]
        if failed:
            logger.warning(f"Type mismatches found in columns: {failed}")
        else:
            logger.info("All type checks passed.")
