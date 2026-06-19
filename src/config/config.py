import os
import torch

# Dataset
DATA_FLAG = 'chestmnist'
DOWNLOAD = True

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Artifact directories
REPORTS_DIR = 'reports'
TRAINED_MODELS_DIR = 'trained_models'

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# Artifact paths
FINAL_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'best_densenet121_final.pth')
OPTUNA_PARAMS_PATH = os.path.join(REPORTS_DIR, 'optuna_best_params.json')
FINAL_METRICS_PATH = os.path.join(REPORTS_DIR, 'final_metrics.json')
