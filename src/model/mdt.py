from typing import Any

import pandas as pd
from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    SelectBySingleFeaturePerformance,
)
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Create target column from stars
def transform_stars_to_target(df: pd.DataFrame, stars_column: str) -> pd.DataFrame:
    """
    Transform the stars column into a categorical target column:
    - 0 if stars <= 2
    - 1 if stars == 3
    - 2 if stars >= 4

    Args:
        df (pd.DataFrame): Input DataFrame with a 'stars' column.
        stars_column (str): Name of the column to transform.

    Returns:
        pd.DataFrame: DataFrame with a new 'target' column and without the original stars column.
    """
    logger.info("Transforming stars column into target column.")
    df = df.copy()

    # Step 1: define criteria for classification
    def classify(stars: float) -> int:
        NEGATIVE_THRESHOLD = 2
        NEUTRAL_VALUE = 3
        if stars <= NEGATIVE_THRESHOLD:
            return 0
        if stars == NEUTRAL_VALUE:
            return 1
        return 2

    # Step 2: apply classification
    df["target"] = df[stars_column].apply(classify)
    return df.drop(columns=[stars_column])


# Split data into train and test sets
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train/test sets based on a target column.

    Args:
        df: Input DataFrame.
        target_column: Column to use as the target.
        test_size: Proportion of data to use for testing.
        random_state: Seed for reproducibility.

    Returns:
        tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into train and test sets.")
    # Step 1: check if the target column exists
    df = df.copy()
    y = df.pop(target_column)
    X = df

    # Step 2: check if the target column is categorical
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


class EncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes categorical and boolean features using OneHotEncoder.
    """

    def __init__(self, drop_first: bool = True) -> None:
        self.drop_first = drop_first
        self.cat_cols: list[str] = []
        self.bool_cols: list[str] = []
        self.ohe: OneHotEncoder | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EncodingTransformer":
        self.cat_cols = X.select_dtypes(
            include=["category", "object", "string"]
        ).columns.tolist()
        self.bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

        logger.info("Fitting EncodingTransformer.")
        self.ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first" if self.drop_first else None,
        )
        self.ohe.fit(X[self.cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ohe is None:
            msg = "EncodingTransformer must be fitted before calling transform()."
            raise RuntimeError(msg)
        assert self.ohe is not None

        logger.info("Transforming data with EncodingTransformer.")
        X = X.copy()
        cat_encoded = self.ohe.transform(X[self.cat_cols])
        cat_feature_names = self.ohe.get_feature_names_out(self.cat_cols)
        bool_encoded = X[self.bool_cols].astype("int8")
        passthrough = X.drop(columns=self.cat_cols + self.bool_cols)

        return pd.concat(
            [
                pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index),
                bool_encoded,
                passthrough,
            ],
            axis=1,
        )


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using the group-wise mean of the target variable during fit.
    At transform time, if the class label is not available, falls back to the global mean.
    """

    def __init__(self, columns_imputer: list[str]) -> None:
        self.columns_imputer = columns_imputer
        self.group_means_: dict[str, dict[Any, float]] = {}
        self.global_means_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupMeanImputer":
        logger.info("Fitting GroupMeanImputer.")
        for col in self.columns_imputer:
            self.group_means_[col] = X[col].groupby(y).mean().to_dict()
            self.global_means_[col] = X[col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.group_means_ or not self.global_means_:
            msg = "Must call fit() before transform()."
            raise RuntimeError(msg)

        logger.info("Transforming data with GroupMeanImputer.")
        X = X.copy()
        for col in self.columns_imputer:
            X[col] = X[col].fillna(self.global_means_[col])
        return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scales numeric columns using MinMaxScaler.
    """

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ScalerTransformer":
        self.columns_ = X.select_dtypes(
            include=["int32", "int64", "float64"]
        ).columns.tolist()
        logger.info("Fitting ScalerTransformer.")
        self.scaler.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_:
            msg = "ScalerTransformer must be fitted before transform."
            raise RuntimeError(msg)
        assert self.columns_ is not None

        logger.info("Transforming data with ScalerTransformer.")
        X = X.copy()
        X[self.columns_] = self.scaler.transform(X[self.columns_])
        return X


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduces dimensionality using a composed pipeline:
    - Drop constant features
    - Drop highly correlated features
    - Univariate feature selection based on model performance
    """

    def __init__(
        self,
        corr_threshold: float,
        importance_threshold: float,
        scoring: str,
        random_state: int,
    ) -> None:
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.scoring = scoring
        self.random_state = random_state

        self.pipeline: Pipeline | None = None
        self.selected_columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DimensionalityReducer":
        """
        Fit the dimensionality reduction pipeline and store the final selected feature names.
        """

        logger.info("Fitting DimensionalityReducer.")

        # 1. Pipeline dimensionality reduction
        self.pipeline = Pipeline(
            [
                ("drop_constant", DropConstantFeatures()),
                (
                    "drop_correlated",
                    DropCorrelatedFeatures(threshold=self.corr_threshold),
                ),
                (
                    "target_selector",
                    SelectBySingleFeaturePerformance(
                        estimator=RandomForestClassifier(
                            n_estimators=50,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                        scoring=self.scoring,
                        cv=3,
                        threshold=self.importance_threshold,
                    ),
                ),
            ]
        )

        # 2.Fit the pipeline
        self.pipeline.fit(X, y)

        # 3. Get the input columns
        intermediate_pipeline = Pipeline(self.pipeline.steps[:-1])
        X_intermediate = intermediate_pipeline.transform(X)

        if isinstance(X_intermediate, pd.DataFrame):
            input_columns = X_intermediate.columns
        else:
            input_columns = [f"feature_{i}" for i in range(X_intermediate.shape[1])]

        # 4. Extract mask and apply it correctly
        selector = self.pipeline.named_steps["target_selector"]
        mask = selector.get_support()
        if len(mask) != len(input_columns):
            msg = (
                f"Column mismatch: selector mask size {len(mask)} "
                f"≠ input column size {len(input_columns)}"
            )
            raise ValueError(msg)

        self.selected_columns_ = [col for col, keep in zip(input_columns, mask) if keep]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None or self.selected_columns_ is None:
            msg = "DimensionalityReducer must be fitted before calling transform()."
            raise RuntimeError(msg)
        assert self.pipeline is not None

        logger.info("Transforming data with DimensionalityReducer.")
        X_reduced = self.pipeline.transform(X)
        return pd.DataFrame(X_reduced, columns=self.selected_columns_, index=X.index)


class MDTYelpData(BaseEstimator, TransformerMixin):
    """
    Apply model-dependent transformations for classification:
    - Imputation by group mean (only if missing values exist)
    - One-hot encoding
    - Scaling
    - Dimensionality reduction
    """

    def __init__(
        self,
        corr_threshold: float,
        importance_threshold: float,
        scoring: str,
        random_state: int,
        target_column: str,
    ) -> None:
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.scoring = scoring
        self.random_state = random_state
        self.target_column = target_column

        self.imputer: GroupMeanImputer | None = None
        self.encoder: EncodingTransformer | None = None
        self.scaler: ScalerTransformer | None = None
        self.reducer: DimensionalityReducer | None = None
        self.skip_imputation: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MDTYelpData":
        logger.info("Fitting MDTYelpData...")

        # Imputation (only if needed)
        cols_with_na = X.columns[X.isnull().any()].tolist()
        if cols_with_na:
            self.imputer = GroupMeanImputer(columns_imputer=cols_with_na)
            X = self.imputer.fit(X, y).transform(X)
        else:
            logger.info("No missing values detected. Skipping imputation.")
            self.skip_imputation = True

        # Encoding
        self.encoder = EncodingTransformer()
        X = self.encoder.fit(X).transform(X)

        # Scaling
        self.scaler = ScalerTransformer()
        X = self.scaler.fit(X).transform(X)

        # Dimensionality Reduction
        self.reducer = DimensionalityReducer(
            corr_threshold=self.corr_threshold,
            importance_threshold=self.importance_threshold,
            scoring=self.scoring,
            random_state=self.random_state,
        )
        self.reducer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not all([self.encoder, self.scaler, self.reducer]):
            msg = "MDTYelpData must be fitted before calling transform()."
            raise RuntimeError(msg)

        assert self.encoder is not None
        assert self.scaler is not None
        assert self.reducer is not None

        logger.info("Transforming data with MDTYelpData...")

        if not self.skip_imputation and self.imputer is not None:
            X = self.imputer.transform(X)

        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        X = self.reducer.transform(X)
        return X
