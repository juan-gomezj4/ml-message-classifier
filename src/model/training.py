import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class TrainModelTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that trains a model with provided constructor and parameters.
    `transform` returns the input unchanged.
    """

    def __init__(
        self,
        classifier_fn: Callable[..., Any],
        best_params: dict[str, Any],
    ) -> None:
        self.classifier_fn = classifier_fn
        self.best_params = best_params
        self.model_: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainModelTransformer":
        self.model_ = self.classifier_fn(**self.best_params)
        logger.info(f"Training model with parameters: {self.best_params}")
        self.model_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("TrainModelTransformer must be fitted before calling transform().")
        return X
