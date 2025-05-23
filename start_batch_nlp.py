import argparse
from pathlib import Path
from typing import Callable

from loguru import logger
from omegaconf import OmegaConf

from src.pipelines.feature_pipeline.feature_pipeline import run_feature_pipeline
from src.pipelines.inference_pipeline.inference_pipeline import run_inference_pipeline
from src.pipelines.training_pipeline.training_pipeline import run_training_pipeline

# Base directory
BASE_DIR = Path(__file__).resolve().parents[0]
config_inference = OmegaConf.load(BASE_DIR / "conf/model_inference/inference.yml")


def main(stage: str) -> None:
    """
    Run the specified pipeline stage.

    Args:
        stage: Pipeline stage to run (F=Feature, T=Training, I=Inference, FTI=All)
    """
    stage = stage.upper().strip()
    STAGE_FUNCTIONS: dict[str, list[Callable[[], None]]] = {
        "F": [run_feature_pipeline],
        "T": [lambda: run_training_pipeline()],
        "I": [lambda: run_inference_pipeline()],
        "FTI": [
            run_feature_pipeline,
            lambda: run_training_pipeline(),
            lambda: run_inference_pipeline(),
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
