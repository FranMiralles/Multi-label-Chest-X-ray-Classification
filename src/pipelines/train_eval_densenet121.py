"""Full experiment pipeline: hparam search → training → evaluation → Grad-CAM.

    python -m src.pipelines.pipeline_final
"""

from src.entity.schemas import OptunaConfig, TrainingConfig
from src.models.hparam import run_hparam_search
from src.models.trainer import run_training
from src.models.evaluator import run_evaluation
from src.models.gradcam import run_gradcam
from src.utils.log import log_run


OPTUNA_CFG = OptunaConfig(n_trials=25, n_epochs=8, patience=3)
TRAIN_CFG  = TrainingConfig(n_epochs=30, patience=5)


def main(
    optuna_cfg: OptunaConfig = OPTUNA_CFG,
    train_cfg: TrainingConfig = TRAIN_CFG,
) -> dict:
    """Ejecuta el pipeline completo y devuelve las métricas finales."""
    best_params = run_hparam_search(optuna_cfg)

    model, history, cfg = run_training(best_params, train_cfg)

    final_metrics, optimal_thresholds, *_ = run_evaluation(model, cfg)


    return final_metrics


if __name__ == '__main__':
    with log_run("pipeline_final"):
        main()
