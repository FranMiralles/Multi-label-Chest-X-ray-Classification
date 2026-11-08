# ChestMNIST — Multi-label Chest X-ray Classification

Multi-label classification on the [ChestMNIST](https://medmnist.com/) dataset (14 pathologies) using a fine-tuned DenseNet-121 with Asymmetric Loss, Optuna hyperparameter search, and Grad-CAM++ explainability.

## Problem

ChestMNIST is a multi-label benchmark derived from NIH ChestX-ray14. Each image can simultaneously present zero or more of 14 thoracic pathologies (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia). The dataset is highly imbalanced and multi-label, which makes standard cross-entropy and fixed thresholds suboptimal.

## Approach

| Component | Choice |
|---|---|
| Architecture | DenseNet-121 pretrained on ImageNet, classifier replaced for 14 outputs |
| Loss | Asymmetric Loss (ASL) — down-weights easy negatives, focuses on positives |
| Augmentation | Random rotation, affine translate, horizontal flip + ImageNet normalization |
| HPO | Optuna (TPE sampler, Median pruner) — tunes LR, batch size, ASL γ, scheduler, DA params |
| Threshold calibration | Per-class optimal threshold selected on validation set (Youden, F1-macro, F1-per-class) |
| Explainability | Grad-CAM++ on `denseblock4` — 7 panels covering easy/hard classes, FP/FN, multi-label cases |

## Project structure

```
ChestMNIST/
├── notebooks/
│   └── presentation.pdf                   # Slides of the work
├── notebooks/
│   └── eda/                   # Exploratory Data Analysis
├── results/
│   ├── logs/                  # Execution logs (timestamped .txt per run)
│   ├── reports/               # Plots, metrics JSON, Optuna params
│   └── trained_models/        # Saved model weights (.pth) — gitignored
├── src/
│   ├── config/
│   │   └── config.py          # All paths anchored to PROJECT_ROOT — safe from any CWD
│   ├── data/
│   │   ├── dataset.py         # get_mnist_dataset
│   │   └── dataloader.py      # get_dataloaders / get_val_loader / get_test_loader
│   ├── eda/
│   │   └── visualize.py       # EDA plots saved to results/reports/
│   ├── entity/
│   │   └── schemas.py         # OptunaConfig, TrainingConfig dataclasses
│   ├── features/
│   │   └── transforms.py      # Train / val transform pipelines
│   ├── models/
│   │   ├── architecture.py    # build_densenet121
│   │   ├── losses.py          # AsymmetricLoss
│   │   ├── trainer.py         # EarlyStopping, epoch loops, run_training
│   │   ├── evaluator.py       # Threshold calibration, test metrics, run_evaluation
│   │   ├── hparam.py          # Optuna objective + run_hparam_search
│   │   └── gradcam.py         # Grad-CAM++ panels + run_gradcam
│   ├── pipelines/
│   │   ├── hparam_search.py   # Entry point: python -m src.pipelines.hparam_search
│   │   ├── train_pipeline.py  # Entry point: python -m src.pipelines.train_pipeline
│   │   ├── eval_pipeline.py   # Entry point: python -m src.pipelines.eval_pipeline
│   │   ├── gradcam_pipeline.py
│   │   └── pipeline_final.py  # Full experiment — orchestrates all steps
│   └── utils/
│       ├── plotting.py        # Reusable plot functions (ROC, confusion matrix, training curves)
│       └── log.py             # Stdout tee → results/logs/<name>_<timestamp>.txt
├── pyproject.toml             # Package definition + direct dependencies
└── requirements.txt           # Exact pip freeze for reproducible installs
```

## Installation

### Prerequisites

- Python 3.10+
- Git

### CPU

```bash
git clone <repo-url>
cd ChestMNIST

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # exact versions
pip install -e .                   # register src as importable package
```

### GPU (CUDA 12.x)

Install PyTorch with CUDA wheels first, then the rest:

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

> `pip install -e .` is required once per environment so that `from src.*` imports work regardless of the working directory (notebooks, scripts, CLI).

## Usage

Each pipeline step can be run independently or all at once.

### Full experiment

```bash
python -m src.pipelines.pipeline_final
```

Runs hparam search → training → evaluation → Grad-CAM in sequence. Logs go to `results/logs/pipeline_final_<timestamp>.txt`.

### Individual steps

```bash
# 1. Hyperparameter search (Optuna)
python -m src.pipelines.hparam_search
# → results/reports/optuna_best_params.json
# → results/reports/optuna_search.png

# 2. Train with best params
python -m src.pipelines.train_pipeline
# → results/trained_models/best_densenet121_final.pth
# → results/reports/training_curves.png

# 3. Evaluate on test set
python -m src.pipelines.eval_pipeline
# → results/reports/final_metrics.json
# → results/reports/roc_curves_test.png
# → results/reports/confusion_and_metrics.png

# 4. Grad-CAM++ explainability panels
python -m src.pipelines.gradcam_pipeline
# → results/reports/gradcam_*.png
```

### EDA (notebook)

Open `notebooks/eda/chestmnist.ipynb`. With the venv active and `pip install -e .` done, all `src.*` imports resolve correctly from the notebook.

## Configuration

Experiment parameters are defined in `src/entity/schemas.py` and can be overridden in `pipeline_final.py`:

```python
OPTUNA_CFG = OptunaConfig(
    n_trials=25,   # number of Optuna trials
    n_epochs=8,    # epochs per trial
    patience=3,    # early stopping patience per trial
)

TRAIN_CFG = TrainingConfig(
    n_epochs=30,   # max epochs for final training
    patience=5,    # early stopping patience
)
```

All output paths are declared in `src/config/config.py` and resolved relative to the project root — they work correctly regardless of where scripts are invoked from.
