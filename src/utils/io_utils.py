from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd
from loguru import logger


# Save DataFrame to parquet file
def save_if_needed(
    data: Union[pd.DataFrame, pd.Series], output_path: Union[str, Path, None]
) -> None:
    """
    Save a DataFrame or Series to a parquet file if output_path is provided.

    Args:
        data (pd.DataFrame | pd.Series): Data to save.
        output_path (str | Path | None): Path to save the parquet file. If None, the function does nothing.
    """
    if output_path:
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.rename("target")
            data = data.to_frame()
        logger.info(f"Saving data to {output_path}")
        data.to_parquet(output_path, index=False)


# Save pipeline to joblib file
def save_pipeline_if_needed(pipeline: Any, output_path: Union[str, Path, None]) -> None:
    """
    Save the pipeline to a file if output_path is provided.

    Args:
        pipeline (Any): The pipeline to save.
        output_path (Union[str, Path, None]): Path to save the pipeline. If None, the function does nothing.
    """
    if output_path:
        logger.info(f"Saving pipeline to {output_path}")
        joblib.dump(pipeline, output_path)


# Load data from parquet file
def read_input_data(data_path: Path) -> pd.DataFrame:
    """
    Read input data for inference from a Parquet file.

    Args:
        data_path (Path): Path to the input data (.parquet).

    Returns:
        pd.DataFrame: Loaded input data.
    """
    return pd.read_parquet(data_path)


# Load model from joblib file
def load_model(model_path: Path) -> Any:
    """
    Load the trained model from a serialized file (.pkl or .joblib).

    Args:
        model_path (Path): Path to the model file.

    Returns:
        Any: Loaded model object.
    """
    return joblib.load(model_path)
