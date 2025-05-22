from collections.abc import Callable
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.utils.validation import check_is_fitted
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
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainModelTransformer":
        logger.info(f"Training model with parameters: {self.best_params}")
        self.model_ = self.classifier_fn(**self.best_params)
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "model_")
        if not self.is_fitted_:
            raise RuntimeError("TrainModelTransformer must be fitted before calling transform().")
        return X
