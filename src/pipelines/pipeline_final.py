"""Experimento DenseNet-121 sobre ChestMNIST.

Punto de entrada para ejecución desatendida:
    python -m src.pipelines.pipeline_final

Los parámetros del experimento (n_trials, n_epochs, patience) se definen
en los dataclasses de entity/schemas.py y se instancian aquí. Para un
experimento alternativo basta crear otro pipeline_*.py con distintos configs.
"""

from src.entity.schemas import OptunaConfig, TrainingConfig
from src.pipelines.hparam_search import run_hparam_search
from src.pipelines.train_pipeline import run_training
from src.pipelines.eval_pipeline import run_evaluation
from src.pipelines.gradcam_pipeline import run_gradcam


# ── Configuración del experimento ─────────────────────────────────────────────
# Modifica aquí para reproducir con distintos valores sin tocar config.py

OPTUNA_CFG = OptunaConfig(
    n_trials=25,
    n_epochs=8,
    patience=3,
)

TRAIN_CFG = TrainingConfig(
    n_epochs=30,
    patience=5,
)


# ── Orquestación ──────────────────────────────────────────────────────────────

def main(
    optuna_cfg: OptunaConfig = OPTUNA_CFG,
    train_cfg: TrainingConfig = TRAIN_CFG,
    skip_gradcam: bool = False,
) -> dict:
    """Ejecuta el pipeline completo y devuelve las métricas finales."""
    best_params = run_hparam_search(optuna_cfg)

    model, history, cfg = run_training(best_params, train_cfg)

    final_metrics, optimal_thresholds, *_ = run_evaluation(model, cfg)

    if not skip_gradcam:
        run_gradcam(model, optimal_thresholds)

    return final_metrics


if __name__ == '__main__':
    main()
