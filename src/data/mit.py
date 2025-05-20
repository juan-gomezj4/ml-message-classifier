from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class DuplicateRemoverTransformer(BaseEstimator, TransformerMixin):
    """
    Remove duplicate rows from the DataFrame.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame) -> "DuplicateRemoverTransformer":
        """
        No fitting is needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            DuplicateRemoverTransformer: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with duplicate rows removed.
        """
        logger.info("Removing duplicate rows from the DataFrame.")
        return X.drop_duplicates()


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Clean text by lowercasing, removing punctuation and extra spaces.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def fit(self, X: pd.DataFrame) -> "TextCleanerTransformer":
        """
        No fitting is needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            TextCleanerTransformer: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text by lowercasing, removing punctuation and extra spaces.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with cleaned text.
        """
        # Step 1: copy the DataFrame
        X = X.copy()

        # Step 2: Clean the
        logger.info(f"Cleaning the text column: {self.text}")
        X[f"{self.text}_clean"] = (
            X[self.text]
            .fillna("")
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)  # remove punctuation
            .str.replace(r"\s+", " ", regex=True)  # normalize spaces
            .str.strip()
        ).astype("object")

        # Step 3: drop original text column
        X.drop(columns=[self.text], inplace=True, errors="ignore")
        return X


class CategoryStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate mean, std, and relative value for a numeric column grouped by a category column.
    """

    def __init__(self, group_col: str, value_col: str) -> None:
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X: pd.DataFrame) -> "CategoryStatsTransformer":
        """
        Fit the transformer by calculating the mean and std of the value column grouped by the group column.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            CategoryStatsTransformer: Returns self.
        """
        # Step 1: copy the DataFrame
        X = X.copy()
        # Step 2: convert to numeric to avoid type issues
        X_valid = X[[self.group_col, self.value_col]].dropna()

        # Step 3: calculate mean and std
        self.group_mean = X_valid.groupby(self.group_col, observed=True)[self.value_col].mean()
        self.group_std = (
            X_valid.groupby(self.group_col, observed=True)[self.value_col].std().fillna(0)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by adding mean, std, and relative value columns.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with added mean, std, and relative value columns.
        """
        # Step 1: copy the DataFrame
        logger.info("Transforming the DataFrame by adding mean, std, and relative value columns.")
        X = X.copy()

        # Convert to numeric to avoid type issues
        X[self.value_col] = pd.to_numeric(X[self.value_col], errors="coerce")
        avg = pd.to_numeric(X[self.group_col].map(self.group_mean), errors="coerce")
        std = pd.to_numeric(X[self.group_col].map(self.group_std), errors="coerce")

        # Add aggregated columns
        X[f"{self.value_col}_avg_by_{self.group_col}"] = avg.astype("float32")
        X[f"{self.value_col}_std_by_{self.group_col}"] = std.astype("float32")
        X[f"{self.value_col}_relative_to_avg"] = (X[self.value_col] - avg).astype("float32")

        return X


class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Generate text embeddings using a pretrained model.
    """

    def __init__(
        self,
        text_clean: str,
        model_name: str,
        n_components: int | None = None,
    ) -> None:
        self.text_clean = text_clean
        self.model_name = model_name
        self.n_components = n_components

    def fit(self, X: pd.DataFrame) -> "TextEmbeddingTransformer":
        """
        Fit the transformer by loading the embedding model.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            TextEmbeddingTransformer: Returns self.
        """
        # Step 1: copy the DataFrame
        X = X.copy()

        # Step 2: Load embedding model
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by generating text embeddings.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with added embedding columns.
        """
        logger.info("Transforming the DataFrame by generating text embeddings.")
        # Step 1: copy the DataFrame
        X = X.copy()

        # Step 2: Check column not null
        texts = X[self.text_clean].fillna("").tolist()

        # Step 3: Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Step 4: Optionally reduce dimensionality
        if self.n_components is not None:
            embeddings = embeddings[:, : self.n_components]

        # Step 5: Create DataFrame for embedding columns
        emb_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=X.index)
        return pd.concat([X, emb_df], axis=1)


class MITYelpData(BaseEstimator, TransformerMixin):
    """
    Apply model-independent transformations to Yelp data.
    """

    def __init__(
        self,
        text_column: str,
        group_col: str,
        value_col: str,
        embedding_model: str,
        n_components: int | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """
        Initialize MITYelpData with transformation parameters.

        Args:
            text_column (str): Column name of the raw text.
            group_col (str): Column name to group by (e.g., category).
            value_col (str): Column name with numeric values (e.g., text length).
            embedding_model (str): Name of the model to use for embeddings.
            n_components (Optional[int]): Number of embedding components to keep.
            output_path (Optional[Union[str, Path]]): Optional path to save transformed data.
        """
        self.text_column = text_column
        self.group_col = group_col
        self.value_col = value_col
        self.embedding_model = embedding_model
        self.n_components = n_components
        self.output_path = Path(output_path) if output_path else None

        # List of transformers to apply in sequence
        self.transformers: list[BaseEstimator] = [
            DuplicateRemoverTransformer(),
            TextCleanerTransformer(text=self.text_column),
            CategoryStatsTransformer(group_col=self.group_col, value_col=self.value_col),
            TextEmbeddingTransformer(
                text_clean=f"{self.text_column}_clean",
                model_name=self.embedding_model,
                n_components=self.n_components,
            ),
        ]

    def fit(self, X: pd.DataFrame) -> "MITYelpData":
        """
        Fit each transformer in sequence.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            MITYelpData: self
        """
        for transformer in self.transformers:
            transformer.fit(X)
        return self

    def _save_if_needed(self, df: pd.DataFrame) -> None:
        """
        Save the transformed DataFrame to parquet if output path is specified.

        Args:
            df (pd.DataFrame): DataFrame to save.
        """
        if self.output_path:
            logger.info(f"Saving MIT-transformed data to {self.output_path}")
            df.to_parquet(self.output_path, index=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all MIT transformers to the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Applying MIT transformations")
        X_transformed = X.copy()

        for transformer in self.transformers:
            logger.debug("Applying transformer: {}", transformer.__class__.__name__)
            X_transformed = transformer.transform(X_transformed)

        self._save_if_needed(X_transformed)
        logger.info("MIT transformation complete")
        return X_transformed

    def set_output(self, *, transform: Any | None = None) -> "MITYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.
        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            MITYelpData: self
        """
        return self
