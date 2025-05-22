import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import joblib

from src.model.mdt import transform_stars_to_target
from src.pipelines.feature_pipeline.feature_pipeline import run_feature_pipeline
from src.pipelines.training_pipeline.training_pipeline import run_training_pipeline
from src.pipelines.inference_pipeline.inference_pipeline import run_inference_pipeline

# Base directory
BASE_DIR = Path(__file__).resolve().parents[0]
config_feature = OmegaConf.load(BASE_DIR / "conf/data_feature/feature.yml")
config_inference = OmegaConf.load(BASE_DIR / "conf/model_inference/inference.yml")


def run_training_stage() -> None:
    """
    Run the training pipeline: create target variable and train the model.
    """
    logger.info("📥 Loading features...")
    df_feature = pd.read_parquet(config_feature.mit_output_path)

    logger.info("🎯 Generating target variable...")
    df = transform_stars_to_target(df_feature, "stars")

    logger.info("🏃 Running training pipeline...")
    run_training_pipeline(data=df)


def run_inference_stage() -> None:
    """
    Run the inference pipeline: load data and model, apply transformations, and predict.
    """
    logger.info("📥 Loading data and model")
    df = pd.read_parquet(config_inference.paths.new_data)
    pipeline = joblib.load(config_inference.paths.best_model)

    logger.info("🏃 Running inference pipeline...")
    run_inference_pipeline(data=df, model=pipeline)


def main(stage: str) -> None:
    """
    Run the specified pipeline stage.

    Args:
        stage: Pipeline stage to run (F=Feature, T=Training, I=Inference, FTI=All)
    """
    stage = stage.upper().strip()
    STAGE_FUNCTIONS: dict[str, list[Callable[[], None]]] = {
        "F": [run_feature_pipeline],
        "T": [run_training_stage],
        "I": [run_inference_stage],
        "FTI": [
            run_feature_pipeline,
            run_training_stage,
            run_inference_stage,
        ],
    }

    if stage not in STAGE_FUNCTIONS:
        logger.error(f"❌ Stage '{stage}' not recognized. Use F, T, I or FTI.")
        return

    logger.info(f"🚀 Running stage: {stage}")
    for step_fn in STAGE_FUNCTIONS[stage]:
        step_fn()
    logger.success(f"✅ Stage {stage} completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline stages")
    parser.add_argument(
        "--stage",
        type=str,
        default="FTI",
        help="Stage to run: F (Feature), T (Training), I (Inference), FTI (all)",
    )
    args = parser.parse_args()

    try:
        main(args.stage)
    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
