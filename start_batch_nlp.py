import argparse
from pathlib import Path

from loguru import logger

from src.pipelines.feature_pipeline.feature_pipeline import run_feature_pipeline
from src.pipelines.training_pipeline.training_pipeline import run_training_pipeline

# Base directory
BASE_DIR = Path(__file__).resolve().parents[0]


def main(stage: str) -> None:
    stage = stage.upper()

    STAGE_FUNCTIONS = {
        "F": [run_feature_pipeline],
        "T": [run_training_pipeline],
        "I": [lambda: logger.warning("🔧 Etapa I (inferencia) aún no implementada.")],
        "FTI": [
            run_feature_pipeline,
            run_training_pipeline,
            lambda: logger.warning("🔧 Etapa I (inferencia) aún no implementada."),
        ],
    }

    if stage not in STAGE_FUNCTIONS:
        logger.error(f"❌ Etapa '{stage}' no reconocida. Usa F, T, I o FTI.")
        return

    logger.info(f"🔁 Ejecutando etapa: {stage}")
    for step_fn in STAGE_FUNCTIONS[stage]:
        step_fn()

    logger.success(f"✅ Etapa {stage} completada con éxito.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta etapas del pipeline ML")
    parser.add_argument(
        "--stage",
        type=str,
        default="FTI",
        help="Etapa a ejecutar: F (Feature), T (Training), I (Inference), FTI (todo)",
    )
    args = parser.parse_args()

    try:
        main(args.stage)
    except Exception as e:
        logger.error(f"❌ Pipeline falló: {e}")
