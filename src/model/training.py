import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class TrainModelTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that trains a model using the provided classifier function and best parameters.
    The transform method passes the data through unchanged.
    """

    def __init__(
        self,
        classifier_fn: Callable[..., Any],
        best_params: dict[str, Any],
    ) -> None:
        """
        Initialize the transformer.

        Args:
            classifier_fn (Callable[..., Any]): A classifier constructor (e.g., RandomForestClassifier).
            best_params (Dict[str, Any]): Best parameters for the classifier.
        """
        self.classifier_fn = classifier_fn
        self.best_params = best_params
        self.model_: Any | None = None

    @staticmethod
    def load_model_name(model_name_path: str | Path, classifier_registry: Any) -> Any:
        """
        Load classifier constructor by reading model name from a file.

        Args:
            model_name_path: Path to the file containing the model name.
            classifier_registry: Dictionary mapping model names to constructors.

        Returns:
            Classifier constructor function.
        """
        # Read the model name from the file
        logger.info(f"Loading model name from {model_name_path}")
        path = Path(model_name_path)
        model_name = path.read_text().strip()
        try:
            return classifier_registry[model_name]
        except KeyError:
            raise ValueError(f"Model '{model_name}' not found in classifier_registry.")

    @staticmethod
    def load_model_parameters(params_path: str | Path) -> Any:
        """
        Load model parameters from a JSON file.

        Args:
            params_path: Path to JSON file with parameters.

        Returns:
            Dictionary of parameters.
        """
        # Read the parameters from the JSON file
        logger.info(f"Loading model parameters from {params_path}")
        path = Path(params_path)
        return json.loads(path.read_text())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainModelTransformer":
        """
        Fit the classifier to the data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            TrainModelTransformer: self
        """
        # Fit the classifier
        logger.info("Fitting the model")
        self.model_ = self.classifier_fn(**self.best_params)
        self.model_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Pass the data through unchanged.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.DataFrame: Unchanged feature matrix.
        """
        logger.info("Transforming the data")
        return X

    def set_output(self, *, transform: Any | None = None) -> "TrainModelTransformer":
        """
        Method for compatibility with scikit-learn's set_output API.
        Args:
            transform (Optional[Any]): Ignored.

        Returns:
            TrainModelTransformer: self
        """
        return self
