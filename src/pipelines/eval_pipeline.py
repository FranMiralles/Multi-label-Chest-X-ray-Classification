"""Entry point: threshold calibration and test-set evaluation.

    python -m src.pipelines.eval_pipeline
"""

from src.models.evaluator import run_evaluation
from src.utils.log import log_run

if __name__ == "__main__":
    with log_run("eval"):
        run_evaluation()
