from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class ValidateYelpData(BaseEstimator, TransformerMixin):
    """
    Transformer for validating Yelp data structure and types.
    Uses simple type verification instead of schema validation.
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
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize ValidateYelpData.
        Args:
            drop_columns: List of columns to drop.
            drop_columns_na: List of columns to drop if they contain NA values.
            cols_categoric: List of categorical columns.
            cols_numeric_float: List of float columns.
            cols_numeric_int: List of integer columns.
            cols_boolean: List of boolean columns.
            cols_string: List of string columns.
            col_date: Date column.
            output_path: Optional output path for saving the validated data.
        """
        self.drop_columns = drop_columns
        self.drop_columns_na = drop_columns_na
        self.cols_categoric = cols_categoric
        self.cols_numeric_float = cols_numeric_float
        self.cols_numeric_int = cols_numeric_int
        self.cols_boolean = cols_boolean
        self.cols_string = cols_string
        self.col_date = col_date
        self.output_path = Path(output_path) if output_path else None

    def fit(self, X: pd.DataFrame) -> "ValidateYelpData":
        """
        Fit method (no-op). Returns self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            ValidateYelpData: Returns self.
        """
        return self

    # Save dato to parquet if output path is specified
    def _save_if_needed(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to parquet if output path is specified.

        Args:
            df (pd.DataFrame): DataFrame to save.

        Returns:
            None
        """
        if self.output_path:
            logger.info(f"Saving validated data to {self.output_path}")
            df.to_parquet(self.output_path, index=False)

    # Helper function to safely cast columns to a specified dtype
    def _safe_cast(self, df: pd.DataFrame, cols: list[str], dtype: str) -> pd.DataFrame:
        """
        Cast columns to specified dtype if they exist in the DataFrame.
        """
        valid_cols = [col for col in cols if col in df.columns]
        if valid_cols:
            logger.debug(f"Casting to {dtype}: {valid_cols}")
            df[valid_cols] = df[valid_cols].astype(dtype)
        return df

    # Verify data types
    def _verify_types(self, X: pd.DataFrame) -> Dict[str, bool]:
        """
        Verify that columns have the expected types after conversion.

        Args:
            X (pd.DataFrame): DataFrame to verify.

        Returns:
            Dict[str, bool]: Dictionary of type checks with column names as keys
            and boolean values indicating if the check passed.
        """
        type_checks = {}

        type_map = {
            "categoric": (self.cols_categoric, pd.api.types.is_categorical_dtype),
            "float": (self.cols_numeric_float, pd.api.types.is_float_dtype),
            "int": (self.cols_numeric_int, pd.api.types.is_integer_dtype),
            "bool": (self.cols_boolean, pd.api.types.is_bool_dtype),
            "string": (
                self.cols_string,
                lambda col: pd.api.types.is_string_dtype(col)
                or pd.api.types.is_object_dtype(col),
            ),
        }

        for type_name, (columns, check_fn) in type_map.items():
            for col in filter(lambda c: c in X.columns, columns):
                type_checks[f"{col}_is_{type_name}"] = check_fn(X[col])

        # Check datetime column separately
        if self.col_date in X.columns:
            type_checks[f"{self.col_date}_is_datetime"] = (
                pd.api.types.is_datetime64_dtype(X[self.col_date])
            )

        return type_checks

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by dropping unnecessary columns, converting types,
        and validating that types are correctly assigned.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Starting data validation process")

        # 1. Drop unnecessary columns and NA values
        logger.info("Dropping unnecessary columns")
        X = X.drop(columns=self.drop_columns, errors="ignore")
        X = X.dropna(subset=self.drop_columns_na)

        # 2. Convert columns to appropriate types
        logger.info("Converting columns to appropriate data types")
        X = self._safe_cast(X, self.cols_categoric, "category")
        X = self._safe_cast(X, self.cols_numeric_float, "float")
        X = self._safe_cast(X, self.cols_numeric_int, "int32")
        X = self._safe_cast(X, self.cols_boolean, "bool")
        X = self._safe_cast(X, self.cols_string, "object")

        if self.col_date in X.columns:
            X[self.col_date] = pd.to_datetime(X[self.col_date], errors="coerce")
        else:
            logger.warning(f"Missing datetime column: {self.col_date}")

        # 3. Verify types are correctly assigned
        logger.info("Verifying data types")
        type_checks = self._verify_types(X)

        # Log any type issues
        failed_checks = {key: val for key, val in type_checks.items() if val is False}
        if failed_checks:
            logger.warning(f"Type verification failed for: {failed_checks}")
        else:
            logger.info("All type verifications passed")

        # 4. Save if needed
        self._save_if_needed(X)

        logger.info("Data validation completed successfully")
        return X

    def set_output(self, *, transform: Optional[Any] = None) -> "ValidateYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.
        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            ValidateYelpData: self
        """
        return self
