"""Entry point: final model training.

    python -m src.pipelines.train_pipeline
"""

from src.models.trainer import run_training
from src.utils.log import log_run

if __name__ == "__main__":
    with log_run("train"):
        run_training()
