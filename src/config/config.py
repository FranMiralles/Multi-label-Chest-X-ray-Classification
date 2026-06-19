from pathlib import Path
import torch

# Project root — absolute, works regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset
DATA_FLAG = 'chestmnist'
DOWNLOAD = True

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Artifact directories
RESULTS_DIR = PROJECT_ROOT / 'results'
REPORTS_DIR = RESULTS_DIR / 'reports'
TRAINED_MODELS_DIR = RESULTS_DIR / 'trained_models'
LOGS_DIR = RESULTS_DIR / 'logs'

for _d in (REPORTS_DIR, TRAINED_MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Artifact paths
FINAL_MODEL_PATH = TRAINED_MODELS_DIR / 'best_densenet121_final.pth'
OPTUNA_PARAMS_PATH = REPORTS_DIR / 'optuna_best_params.json'
FINAL_METRICS_PATH = REPORTS_DIR / 'final_metrics.json'
