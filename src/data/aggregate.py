import re
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class EliteAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Standardize values in the 'elite' column and count how many elite years each user has.
    """

    def __init__(
        self,
        elite: str,
        elite_count: int,
    ) -> None:
        self.elite = elite
        self.elite_count = elite_count

    def fit(self, X: Any | None = None) -> "EliteAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning elite years")
        X = X.copy()

        # Step 1: fill nulls and split by comma
        elite_split = X[self.elite].fillna("").str.split(",")

        # Step 2: standardize each year format (e.g. '12' -> '2012')
        logger.info("Standardizing elite years")
        YEAR_SHORT_LENGTH = 2
        elite_clean = elite_split.apply(
            lambda lst: sorted(
                {
                    "20" + y
                    if len(y) == YEAR_SHORT_LENGTH and y.isdigit()
                    else y.strip()
                    for y in lst
                }
            )
        )

        # Step 3: join years into a single string
        logger.info("Joining elite years")
        X[self.elite] = elite_clean.str.join(",")

        # Step 4: count how many years each user has
        logger.info("Counting elite years")
        X[self.elite_count] = elite_clean.str.len().astype("int32")

        return X


class FrequencyEncodeAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Replace categorical values with their frequency in the training data.
    """

    def __init__(self, frequency_encode: list[str]) -> None:
        self.frequency_encode = frequency_encode
        self.freq_maps: dict[str, pd.Series] = {}

    def fit(self, X: Any | None = None) -> "FrequencyEncodeAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Frequency encoding categorical columns")
        X = X.copy()
        for col in self.frequency_encode:
            freq_map = X[col].value_counts(normalize=True)
            X[f"{col}_freq"] = X[col].map(freq_map).astype("float64")
        return X


class BinaryFlagAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Create binary flags for numeric columns where value > 0.
    """

    def __init__(self, binary_flag: list[str]) -> None:
        self.binary_flag = binary_flag

    def fit(self, X: Any | None = None) -> "BinaryFlagAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating binary flags for numeric columns")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        for col in self.binary_flag:
            # Step 2: Create binary flag column
            X[f"is_{col}"] = (X[col] > 0).astype(bool)
        return X


class QCutLevelAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Bin numeric columns into quartile levels using qcut.
    """

    def __init__(self, qcut_level: list[str], labels: list[int] | None = None):
        if labels is None:
            labels = [0, 1, 2, 3]
        self.qcut_level = qcut_level
        self.labels = labels

    def fit(self, X: Any | None = None) -> "QCutLevelAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Binning numeric columns into quartile levels")
        # Step 1: Bin numeric columns into quartiles
        X = X.copy()
        for col in self.qcut_level:
            if col not in X.columns:
                msg = f"Column '{col}' not found in DataFrame."
                raise ValueError(msg)
            X[f"{col}_level"] = pd.qcut(
                X[col], q=4, labels=self.labels, duplicates="drop"
            ).astype("int32")
        return X


class FansLevelAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Transform 'fans' column into 3 levels:
    0 = no fans, 1 = low, 2 = high (above 90 percentile).
    """

    def __init__(self, fans: str) -> None:
        self.fans = fans
        self.p90: float = 0.0

    def fit(self, X: Any | None = None) -> "FansLevelAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming fans column into levels")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()

        # Step 2: Define logic using np.select for speed
        p90 = X[self.fans][X[self.fans] > 0].quantile(0.90)
        conditions = [X[self.fans] == 0, X[self.fans] <= p90]
        choices = [0, 1]
        X[f"{self.fans}_level"] = np.select(conditions, choices, default=2).astype(
            "int32"
        )
        return X


class TextLengthAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate how many characters the text has.
    """

    def __init__(self, text: str, text_length: str) -> None:
        self.text = text
        self.text_length = text_length

    def fit(self, X: Any | None = None) -> "TextLengthAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating text length")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Count characters in text
        X[self.text_length] = X[self.text].str.len().fillna(0).astype("int64")
        return X


class WordCountAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Count how many words the text has.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def fit(self, X: Any | None = None) -> "WordCountAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Counting words in text")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Count number of words in text
        X["word_count"] = X[self.text].fillna("").str.split().str.len().astype("int32")
        return X


class ExclamationFlagAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Check if text contains at least one exclamation mark (! or ¡).
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def fit(self, X: Any | None = None) -> "ExclamationFlagAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Checking for exclamation marks in text")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Flag rows where text contains '!' or '¡'
        X["has_exclamation"] = (
            X[self.text].str.contains(r"[!¡]{1,}", regex=True, na=False).astype(bool)
        )
        return X


class CategoryGroupAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Group categories into main groups using regex patterns.
    """

    def __init__(self, categories: str) -> None:
        self.categories = categories
        self.patterns: dict[str, str] = {
            "restaurant": r"\brestaurant|pizza|food|diner\b",
            "bar": r"\bbar|pub|nightlife|cocktail\b",
            "health": r"\bhealth|spa|doctor|clinic|fitness|dentist\b",
        }
        # Compile regex patterns for speed
        self.compiled_patterns = {k: re.compile(v) for k, v in self.patterns.items()}

    def fit(self, X: Any | None = None) -> "CategoryGroupAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Categorizing main category group")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()

        # Step 2: Apply regex matching to assign category group
        def categorize(cats: str) -> str:
            if pd.isna(cats):
                return "other"
            cats = cats.lower()
            for label, pattern in self.compiled_patterns.items():
                if pattern.search(cats):
                    return label
            return "other"

        # Step 3: Apply categorization function to each row
        X["main_category_group"] = (
            X[self.categories].apply(categorize).astype("category")
        )
        return X


class CategoryCountAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Count how many categories each business has.
    """

    def __init__(self, categories: str) -> None:
        self.categories = categories

    def fit(self, X: Any | None = None) -> "CategoryCountAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Counting categories for each business")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Count number of elements in the category list
        X["category_count"] = (
            X[self.categories].fillna("").str.split(",").str.len().astype("int32")
        )
        return X


class DateAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Extract date features (year, month, day of week, weekend, quarter) from a datetime column.
    """

    def __init__(self, date: str) -> None:
        self.date = date

    def fit(self, X: Any | None = None) -> "DateAggregateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting date features")
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()

        logger.info("Converting date column to datetime")
        # Step 2: convert to datetime
        X[self.date] = pd.to_datetime(X[self.date], errors="coerce")

        logger.info("Extracting year from date")
        # Step 3: extract year
        X["review_year"] = X[self.date].dt.year.astype("category")

        logger.info("Extracting month from date")
        # Step 4: extract month
        X["review_month"] = X[self.date].dt.month.astype("category")

        logger.info("Extracting day of week from date")
        # Step 5: extract day of week (0 = Monday, 6 = Sunday)
        X["review_dayofweek"] = X[self.date].dt.dayofweek

        logger.info("Creating weekend flag")
        # Step 6: create weekend flag
        WEEKEND_START_DAY = 5
        X["is_weekend"] = (X["review_dayofweek"] >= WEEKEND_START_DAY).astype("bool")

        logger.info("Re-casting day of week as category")
        # Step 7: re-cast day of week as category
        X["review_dayofweek"] = X["review_dayofweek"].astype("category")

        logger.info("Extracting quarter from date")
        # Step 8: extract quarter
        X["review_quarter"] = X[self.date].dt.quarter.astype("category")

        return X


class AggregateYelpData(BaseEstimator, TransformerMixin):
    """
    Apply a sequence of feature engineering transformations to Yelp data.
    """

    def __init__(
        self,
        elite: str,
        elite_count: int,
        frequency_encode: list[str],
        binary_flag: list[str],
        qcut_level: list[str],
        fans: str,
        text: str,
        text_length: str,
        categories: str,
        date: str,
    ) -> None:
        self.elite = elite
        self.elite_count = elite_count
        self.frequency_encode = frequency_encode
        self.binary_flag = binary_flag
        self.qcut_level = qcut_level
        self.fans = fans
        self.text = text
        self.text_length = text_length
        self.categories = categories
        self.date = date

        self.transformers: list[BaseEstimator] = [
            EliteAggregateTransformer(elite=self.elite, elite_count=self.elite_count),
            FrequencyEncodeAggregateTransformer(frequency_encode=self.frequency_encode),
            BinaryFlagAggregateTransformer(binary_flag=self.binary_flag),
            QCutLevelAggregateTransformer(qcut_level=self.qcut_level),
            FansLevelAggregateTransformer(fans=self.fans),
            TextLengthAggregateTransformer(
                text=self.text, text_length=self.text_length
            ),
            WordCountAggregateTransformer(text=self.text),
            ExclamationFlagAggregateTransformer(text=self.text),
            CategoryGroupAggregateTransformer(categories=self.categories),
            CategoryCountAggregateTransformer(categories=self.categories),
            DateAggregateTransformer(date=self.date),
        ]

    def fit(self, X: Any = None, y: Any = None) -> "AggregateYelpData":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying feature engineering transformations...")

        # Step 1: transform the DataFrames
        X_transformed = X.copy()
        for transformer in self.transformers:
            if not hasattr(transformer, "transform"):
                msg = (
                    f"{transformer.__class__.__name__} does not implement .transform()"
                )
                raise TypeError(msg)
            logger.debug(f"Applying: {transformer.__class__.__name__}")
            X_transformed = transformer.transform(X_transformed)

        logger.success("Feature engineering completed.")
        return X_transformed
