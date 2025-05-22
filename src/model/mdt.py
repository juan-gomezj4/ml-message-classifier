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

from src.utils.io_utils import save_if_needed


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

    def classify(stars: float) -> int:
        NEGATIVE_THRESHOLD = 2
        NEUTRAL_VALUE = 3
        if stars <= NEGATIVE_THRESHOLD:
            return 0
        if stars == NEUTRAL_VALUE:
            return 1
        return 2

    df["target"] = df[stars_column].apply(classify)
    return df.drop(columns=[stars_column])


# Split data into train and test sets
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int = 42,
    output_path_train: str | None = None,
    output_path_test: str | None = None,
    output_path_ytrain: str | None = None,
    output_path_ytest: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets and optionally save them.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name to use as target.
        test_size (float): Test split ratio.
        random_state (int): Random seed.
        output_path_train (str | None): Path to save X_train.
        output_path_test (str | None): Path to save X_test.
        output_path_ytrain (str | None): Path to save y_train.
        output_path_ytest (str | None): Path to save y_test.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Splitting data into train and test sets.")
    df = df.copy()

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Guardar cada división si se especifica el path
    save_if_needed(X_train.reset_index(drop=True), output_path_train)
    save_if_needed(X_test.reset_index(drop=True), output_path_test)
    save_if_needed(y_train.reset_index(drop=True), output_path_ytrain)
    save_if_needed(y_test.reset_index(drop=True), output_path_ytest)

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


class EncodingTransformer(BaseEstimator, TransformerMixin):
    """Encode categorical and boolean features."""

    def __init__(self, drop_first: bool = True) -> None:
        self.drop_first = drop_first
        self.cat_cols: list[str] = []
        self.bool_cols: list[str] = []
        self.ohe: OneHotEncoder | None = None

    def fit(self, X: pd.DataFrame) -> "EncodingTransformer":
        """
        Fit the encoder by learning categories and storing column types.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            EncodingTransformer: The fitted transformer.
        """
        logger.info("Fitting EncodingTransformer.")
        self.cat_cols = X.select_dtypes(
            include=["category", "object", "string"]
        ).columns.tolist()
        self.bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        logger.info(f"Categorical columns: {self.cat_cols}")
        logger.info(f"Boolean columns: {self.bool_cols}")

        # Apply one-hot encoding to categorical columns
        self.ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first" if self.drop_first else None,
        )
        self.ohe.fit(X[self.cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by encoding categorical and boolean columns.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with encoded features.
        """
        logger.info("Transforming data with EncodingTransformer.")

        # Step 1: copy dataframe
        X = X.copy()

        # Step 2: check if the encoder has been fitted
        ENCODER_NOT_FITTED_MSG = "The encoder has not been fitted. Run fit() first."
        if self.ohe is None:
            raise ValueError(ENCODER_NOT_FITTED_MSG)

        # Step 3: apply one-hot encoding to categorical columns
        cat_encoded = self.ohe.transform(X[self.cat_cols])

        # Step 4: get feature names for one-hot encoded columns
        cat_feature_names = self.ohe.get_feature_names_out(self.cat_cols)
        bool_encoded = X[self.bool_cols].astype("int8")
        passthrough_cols = X.drop(columns=self.cat_cols + self.bool_cols)
        df_encoded = pd.concat(
            [
                pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index),
                bool_encoded,
                passthrough_cols,
            ],
            axis=1,
        )
        return df_encoded


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using the mean of the target groups."""

    def __init__(self, columns_imputer: list[str]) -> None:
        self.columns_imputer = columns_imputer
        self.group_means_: dict[str, dict[int | float, float]] = {}
        self.y_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupMeanImputer":
        """
        Fit the imputer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Target variable.

        Returns:
            GroupMeanImputer: self
        """
        logger.info("Fitting GroupMeanImputer.")
        X = X.copy()
        self.y_ = y.reset_index(drop=True)
        self.group_means_ = {
            col: X[col].groupby(self.y_).mean().to_dict()
            for col in self.columns_imputer
        }
        logger.info(f"Columns to impute: {self.columns_imputer}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using the mean of the target groups.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        logger.info("Transforming data with GroupMeanImputer.")
        X = X.copy().reset_index(drop=True)
        y = self.y_
        if y is None:
            ENCODER_NOT_FITTED_MSG = "The encoder has not been fitted. Run fit() first."
            raise ValueError(ENCODER_NOT_FITTED_MSG)
        for col in self.columns_imputer:
            means = self.group_means_[col]
            X[col] = X[col].where(~X[col].isna(), y.map(means))
        return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scale numeric features using MinMaxScaler.
    """

    def __init__(self) -> None:
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ScalerTransformer":
        """
        Fit the MinMaxScaler to numeric columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            ScalerTransformer: self
        """
        logger.info("Fitting ScalerTransformer.")
        X = X.copy()
        self.columns_ = X.select_dtypes(
            include=["int32", "int64", "float64"]
        ).columns.tolist()
        self.scaler.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric columns using the fitted scaler.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with scaled numeric columns.
        """
        logger.info("Transforming data with ScalerTransformer.")

        # Step 1: copy dataframe
        X = X.copy()

        # Step 2: scale numeric columns
        logger.info(f"Scaling columns: {self.columns_}")
        X[self.columns_] = self.scaler.transform(X[self.columns_])
        return X


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduce dimensionality using:
    - Drop constant features
    - Drop highly correlated features
    - Univariate feature selection
    - Sequential forward feature selection
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
        Fit the dimensionality reduction steps.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            DimensionalityReducer: self
        """
        logger.info("Fitting DimensionalityReducer pipeline.")

        # Define pipeline
        self.pipeline = Pipeline(
            steps=[
                ("drop_constant", DropConstantFeatures()),
                (
                    "drop_correlated",
                    DropCorrelatedFeatures(threshold=self.corr_threshold),
                ),
                (
                    "target_selector",
                    SelectBySingleFeaturePerformance(
                        estimator=RandomForestClassifier(
                            n_estimators=50, random_state=self.random_state, n_jobs=-1
                        ),
                        scoring=self.scoring,
                        cv=3,
                        threshold=self.importance_threshold,
                    ),
                ),
            ]
        )

        # Fit pipeline
        self.pipeline.fit(X, y)

        # Store selected feature names
        X_transformed = self.pipeline.transform(X)
        self.selected_columns_ = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        logger.info(f"{len(self.selected_columns_)} features selected.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dimensionality reduction to new data.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed features.
        """
        logger.info("Transforming data with DimensionalityReducer.")

        if self.pipeline is None or self.selected_columns_ is None:
            PIPELINE_NOT_FITTED_MSG = "Pipeline has not been fitted."
            raise RuntimeError(PIPELINE_NOT_FITTED_MSG)

        # Apply the pipeline to transform the data
        logger.info("Transforming data with the fitted pipeline.")
        X_reduced = self.pipeline.transform(X)

        return pd.DataFrame(X_reduced, columns=self.selected_columns_, index=X.index)


class MDTYelpData(BaseEstimator, TransformerMixin):
    """
    Apply model-dependent transformations to Yelp data for classification tasks.

    Includes:
    - Encoding (categorical and boolean)
    - Group mean imputation
    - MinMax scaling
    - Dimensionality reduction
    """

    def __init__(
        self,
        corr_threshold: float,
        importance_threshold: float,
        scoring: str,
        random_state: int,
        target_column: str,
        output_path_fit: str,
        output_path_transformed: str,
    ) -> None:
        """Initialize the MDTYelpData transformer.
        Args:
            corr_threshold (float): Correlation threshold for dropping features.
            importance_threshold (float): Importance threshold for feature selection.
            scoring (str): Scoring metric for feature selection.
            random_state (int): Random state for reproducibility.
            n_components (int): Number of components to select.
            target_column (str): Target column name.
            output_path (str): Path to save the fit data.
            output_path_transformed (str): Path to save the transformed data.
        """
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.scoring = scoring
        self.random_state = random_state
        self.target_column = target_column
        self.output_path_fit = output_path_fit
        self.output_path_transformed = output_path_transformed

        self.encoder: EncodingTransformer | None = None
        self.imputer: GroupMeanImputer | None = None
        self.scaler: ScalerTransformer | None = None
        self.reducer: DimensionalityReducer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MDTYelpData":
        """
        Fit all transformation steps using training data.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable (required for supervised steps).

        Returns:
            MDTYelpData: self
        """
        logger.info("Fitting MDTYelpData...")

        # Step 1: Imputation
        columns_with_na = X.columns[X.isnull().any()].tolist()
        self.imputer = GroupMeanImputer(columns_imputer=columns_with_na)
        self.imputer.fit(X, y)
        X_imputed = self.imputer.transform(X)

        # Step 2: Encoding
        self.encoder = EncodingTransformer()
        self.encoder.fit(X)
        X_encoded = self.encoder.transform(X_imputed)

        # Step 3: Scaling
        self.scaler = ScalerTransformer()
        self.scaler.fit(X_encoded)
        X_scaled = self.scaler.transform(X_encoded)

        # Step 4: Dimensionality Reduction
        self.reducer = DimensionalityReducer(
            corr_threshold=self.corr_threshold,
            importance_threshold=self.importance_threshold,
            scoring=self.scoring,
            random_state=self.random_state,
        )
        self.reducer.fit(X_scaled, y)

        # Step 5: save data
        X_reduced = self.reducer.transform(X_scaled)
        save_if_needed(X_reduced, self.output_path_fit)

        logger.info("MDTYelpData fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to new data.

        Args:
            X (pd.DataFrame): New input data.

        Returns:
            pd.DataFrame: Transformed data.
        """
        logger.info("Transforming data with MDTYelpData...")

        assert self.imputer is not None, "Imputer must be fitted."
        X = self.imputer.transform(X)

        assert self.encoder is not None, "Encoder must be fitted."
        X = self.encoder.transform(X)

        assert self.scaler is not None, "Scaler must be fitted."
        X = self.scaler.transform(X)

        assert self.reducer is not None, "Reducer must be fitted."
        X = self.reducer.transform(X)

        # Save transformed data
        logger.info("Saving transformed data.")
        save_if_needed(X, self.output_path_transformed)

        logger.info("Transformation complete.")
        return X

    def set_output(self, *, transform: Any | None = None) -> "MDTYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.
        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            MDTYelpData: self
        """
        return self
