import re
from pathlib import Path
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

    def fit(self, X: pd.DataFrame) -> "EliteAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            EliteAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean 'elite' values and count how many years are listed.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with cleaned 'elite' values and counts.
        """
        X = X.copy()

        # Step 1: fill nulls and split by comma
        elite_split = X[self.elite].fillna("").str.split(",")

        # Step 2: standardize each year format (e.g. '12' -> '2012')
        elite_clean = elite_split.apply(
            lambda lst: sorted(
                {"20" + y if len(y) == 2 and y.isdigit() else y.strip() for y in lst}
            )
        )

        # Step 3: join years into a single string
        X[self.elite] = elite_clean.str.join(",")

        # Step 4: count how many years each user has
        X[self.elite_count] = elite_clean.str.len().astype("int32")

        return X


class FrequencyEncodeAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Replace categorical values with their frequency in the training data.
    """

    def __init__(self, frequency_encode: list[str]) -> None:
        self.frequency_encode = frequency_encode
        self.freq_maps: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame) -> "FrequencyEncodeAggregateTransformer":
        """
        Calculate how often each category appears (frequency).
        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            FrequencyEncodeAggregateTransformer: self
        """
        for col in self.frequency_encode:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace each category with its frequency.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with frequency-encoded columns.
        """
        X = X.copy()

        # Step: map each value to its frequency using the fitted dictionary
        for col in self.frequency_encode:
            X[f"{col}_freq"] = X[col].map(self.freq_maps[col]).astype("float64")

        return X


class BinaryFlagAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Create binary flags for numeric columns where value > 0.
    """

    def __init__(self, binary_flag: list[str]) -> None:
        self.binary_flag = binary_flag

    def fit(self, X: pd.DataFrame) -> "BinaryFlagAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            BinaryFlagAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For each column, create a flag that is True if value > 0.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with binary flags.
        """
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

    def __init__(self, qcut_level: list[str], labels: list[int] = [0, 1, 2, 3]) -> None:
        self.qcut_level = qcut_level
        self.labels = labels

    def fit(self, X: pd.DataFrame) -> "QCutLevelAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            QCutLevelAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric values to levels (quartiles).

        Args:
            X (pd.DataFrame): DataFrame to transform.
            y (Optional[pd.Series]): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame with quartile levels.
        """
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        for col in self.qcut_level:
            # Step 2: Create new column with quartile level
            X[f"{col}_level"] = pd.qcut(X[col], q=4, labels=self.labels, duplicates="drop").astype(
                "int32"
            )
        return X


class FansLevelAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Transform 'fans' column into 3 levels:
    0 = no fans, 1 = low, 2 = high (above 90 percentile).
    """

    def __init__(self, fans: str) -> None:
        self.fans = fans
        self.p90: float = 0.0

    def fit(self, X: pd.DataFrame) -> "FansLevelAggregateTransformer":
        """
        Calculate 90th percentile for fans > 0.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            FansLevelAggregateTransformer: self
        """
        self.p90 = X[self.fans][X[self.fans] > 0].quantile(0.90)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each user to a fan level: 0, 1, or 2.

        Args:
            X (pd.DataFrame): DataFrame to transform.
            y (Optional[pd.Series]): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame with fan levels.
        """
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Define logic using np.select for speed
        conditions = [X[self.fans] == 0, X[self.fans] <= self.p90]
        choices = [0, 1]
        X[f"{self.fans}_level"] = np.select(conditions, choices, default=2).astype("int32")
        return X


class TextLengthAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate how many characters the text has.
    """

    def __init__(self, text: str, text_length: str) -> None:
        self.text = text
        self.text_length = text_length

    def fit(self, X: pd.DataFrame) -> "TextLengthAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            TextLengthAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the length of the text in characters.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with text length.
        """
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

    def fit(self, X: pd.DataFrame) -> "WordCountAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            WordCountAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of words in the text.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with word count.
        """
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

    def fit(self, X: pd.DataFrame) -> "ExclamationFlagAggregateTransformer":
        """
        No fitting needed. Return self.
        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            ExclamationFlagAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check if text contains at least one exclamation mark (! or ¡).

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with exclamation flag.
        """
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

    def fit(self, X: pd.DataFrame) -> "CategoryGroupAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            CategoryGroupAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize each row based on regex patterns.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with main category group.
        """
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
        X["main_category_group"] = X[self.categories].apply(categorize).astype("category")
        return X


class CategoryCountAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Count how many categories each business has.
    """

    def __init__(self, categories: str) -> None:
        self.categories = categories

    def fit(self, X: pd.DataFrame) -> "CategoryCountAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            CategoryCountAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of categories each business has.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with category count.
        """
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()
        # Step 2: Count number of elements in the category list
        X["category_count"] = X[self.categories].fillna("").str.split(",").str.len().astype("int32")
        return X


class DateAggregateTransformer(BaseEstimator, TransformerMixin):
    """
    Extract date features (year, month, day of week, weekend, quarter) from a datetime column.
    """

    def __init__(self, date: str) -> None:
        self.date = date

    def fit(self, X: pd.DataFrame) -> "DateAggregateTransformer":
        """
        No fitting needed. Return self.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            DateAggregateTransformer: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features from a datetime column.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with date features.
        """
        # Step 1: copy the DataFrame to avoid modifying the original
        X = X.copy()

        # Step 2: convert to datetime
        X[self.date] = pd.to_datetime(X[self.date], errors="coerce")

        # Step 3: extract year
        X["review_year"] = X[self.date].dt.year.astype("category")

        # Step 4: extract month
        X["review_month"] = X[self.date].dt.month.astype("category")

        # Step 5: extract day of week (0 = Monday, 6 = Sunday)
        X["review_dayofweek"] = X[self.date].dt.dayofweek

        # Step 6: create weekend flag
        X["is_weekend"] = (X["review_dayofweek"] >= 5).astype("bool")

        # Step 7: re-cast day of week as category
        X["review_dayofweek"] = X["review_dayofweek"].astype("category")

        # Step 8: extract quarter
        X["review_quarter"] = X[self.date].dt.quarter.astype("category")

        return X


class AggregateYelpData(BaseEstimator, TransformerMixin):
    """
    Aggregate Yelp data by applying multiple feature engineering steps.
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
        output_path: str | Path | None = None,
    ) -> None:
        """
        Initialize AggregateYelpData with feature engineering parameters.
        Args:
            elite (str): Column name for elite status.
            elite_count (int): Column name for elite count.
            frequency_encode (List[str]): List of columns to frequency encode.
            binary_flag (List[str]): List of columns to create binary flags.
            qcut_level (List[str]): List of columns to bin into quartile levels.
            fans (str): Column name for fans.
            text (str): Column name for text data.
            text_length (str): Column name for text length.
            categories (str): Column name for categories.
            date (str): Column name for date.
            output_path (Optional[Union[str, Path]]): Path to save the transformed data.
        """
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
        self.output_path = Path(output_path) if output_path else None

        # Define the list of transformers
        self.transformers: list[BaseEstimator] = [
            EliteAggregateTransformer(elite=self.elite, elite_count=self.elite_count),
            FrequencyEncodeAggregateTransformer(frequency_encode=self.frequency_encode),
            BinaryFlagAggregateTransformer(binary_flag=self.binary_flag),
            QCutLevelAggregateTransformer(qcut_level=self.qcut_level),
            FansLevelAggregateTransformer(fans=self.fans),
            TextLengthAggregateTransformer(text=self.text, text_length=self.text_length),
            WordCountAggregateTransformer(text=self.text),
            ExclamationFlagAggregateTransformer(text=self.text),
            CategoryGroupAggregateTransformer(categories=self.categories),
            CategoryCountAggregateTransformer(categories=self.categories),
            DateAggregateTransformer(date=self.date),
        ]

    def fit(self, X: pd.DataFrame) -> "AggregateYelpData":
        """
        Fit each transformer in sequence.

        Args:
            X (pd.DataFrame): DataFrame to fit.

        Returns:
            AggregateYelpData: self

        """
        for transformer in self.transformers:
            transformer.fit(X)
        return self

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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers to the data in order.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Transforming data with aggregate transformers")
        # Step 1: copy the DataFrame to avoid modifying the original
        X_transformed = X.copy()

        # Step 2: Apply each transformer in sequence
        for transformer in self.transformers:
            X_transformed = transformer.transform(X_transformed)
            logger.debug("Applying transformer: {}", transformer.__class__.__name__)

        # Step 3: Save the transformed DataFrame if output path is specified
        self._save_if_needed(X_transformed)
        logger.info("Data transformation complete")
        return X_transformed

    def set_output(self, *, transform: Any | None = None) -> "AggregateYelpData":
        """
        Method for compatibility with scikit-learn's set_output API.

        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            AggregateYelpData: self
        """
        return self
