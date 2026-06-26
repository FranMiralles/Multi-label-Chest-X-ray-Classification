"""Entry point: Grad-CAM++ explainability panels.

    python -m src.pipelines.gradcam_pipeline
"""

from src.models.evaluator import run_evaluation
from src.models.gradcam import run_gradcam
from src.utils.log import log_run

if __name__ == "__main__":
    with log_run("gradcam"):
        _, optimal_thresholds, *_ = run_evaluation()
        run_gradcam(None, optimal_thresholds)
