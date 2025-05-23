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

    def fit(self, X: Any | None = None) -> "DuplicateRemoverTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicate rows from the DataFrame.")
        return X.drop_duplicates()


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Clean text by lowercasing, removing punctuation and extra spaces.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def fit(self, X: Any | None = None) -> "TextCleanerTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return X.drop(columns=[self.text])


class CategoryStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate mean, std, and relative value for a numeric column grouped by a category column.
    """

    def __init__(self, group_col: str, value_col: str) -> None:
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X: Any | None = None) -> "CategoryStatsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "Transforming the DataFrame by adding mean, std, and relative value columns."
        )
        # Step 1: copy the DataFrame
        X = X.copy()

        # Step 2: Calculate mean and std for the specified group and value columns
        group_stats = (
            X[[self.group_col, self.value_col]]
            .dropna()
            .groupby(self.group_col, observed=True)[self.value_col]
        )
        group_mean = group_stats.transform("mean")
        group_std = group_stats.transform("std").fillna(0)

        # Step 3: Add new columns to the DataFrame
        X[f"{self.value_col}_avg_by_{self.group_col}"] = group_mean.astype("float32")
        X[f"{self.value_col}_std_by_{self.group_col}"] = group_std.astype("float32")
        X[f"{self.value_col}_relative_to_avg"] = (
            X[self.value_col] - group_mean
        ).astype("float32")

        return X


class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Generate text embeddings using a pretrained SentenceTransformer model.
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
        self.model = SentenceTransformer(model_name)

    def fit(
        self, X: Any | None = None, y: Any | None = None
    ) -> "TextEmbeddingTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming the DataFrame by generating text embeddings.")
        # Step 1: copy DataFrame
        X = X.copy()

        # Step 2: Get list of texts, replacing NaNs with empty strings
        texts = X[self.text_clean].fillna("").astype(str).tolist()

        # Step 3: Encode texts into embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Step 4: Optionally reduce dimensions
        if self.n_components is not None:
            embeddings = embeddings[:, : self.n_components]

        # Step 5: Build embedding DataFrame with aligned index
        emb_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=X.index)

        # Step 6: Drop original text column and concatenate embeddings
        X.drop(columns=[self.text_clean], inplace=True)
        return pd.concat([X, emb_df], axis=1)


class MITYelpData(BaseEstimator, TransformerMixin):
    """
    Applies a sequence of model-independent transformations to Yelp data.
    """

    def __init__(
        self,
        text_column: str,
        group_col: str,
        value_col: str,
        embedding_model: str,
        n_components: int | None = None,
    ) -> None:
        self.text_column = text_column
        self.group_col = group_col
        self.value_col = value_col
        self.embedding_model = embedding_model
        self.n_components = n_components

        self.transformers: list[BaseEstimator] = [
            DuplicateRemoverTransformer(),
            TextCleanerTransformer(text=self.text_column),
            CategoryStatsTransformer(
                group_col=self.group_col, value_col=self.value_col
            ),
            TextEmbeddingTransformer(
                text_clean=f"{self.text_column}_clean",
                model_name=self.embedding_model,
                n_components=self.n_components,
            ),
        ]

    def fit(self, X: Any = None, y: Any = None) -> "MITYelpData":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting model-independent transformations (MIT)")
        # Step 1: Check if transformers are fitted
        X_transformed = X.copy()
        for transformer in self.transformers:
            if not hasattr(transformer, "transform"):
                msg = f"{transformer.__class__.__name__} must implement .transform()"
                raise TypeError(msg)
            logger.debug(f"Applying: {transformer.__class__.__name__}")
            X_transformed = transformer.transform(X_transformed)

        logger.success("MIT transformations completed.")
        return X_transformed
