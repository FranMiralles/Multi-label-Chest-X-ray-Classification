"""Optuna hyperparameter search logic for DenseNet-121 on ChestMNIST."""

import json
import time

import optuna
import torch.optim as optim
from medmnist import INFO, Evaluator

from src.config.config import DATA_FLAG, DEVICE, OPTUNA_PARAMS_PATH
from src.data.dataloader import get_dataloaders
from src.entity.schemas import OptunaConfig
from src.features.transforms import get_transforms
from src.models.architecture import build_densenet121 as build_model
from src.models.losses import AsymmetricLoss
from src.models.trainer import run_one_epoch_train, run_one_epoch_val
from src.utils.plotting import plot_optuna_search

info = INFO[DATA_FLAG]
N_CLASSES = len(info['label'])


def _objective(trial: optuna.Trial, evaluator_val: Evaluator,
               cfg: OptunaConfig) -> float:
    print(f"\n{'─'*55}")
    print(f"  TRIAL {trial.number + 1}/{cfg.n_trials} — iniciando")
    print(f"{'─'*55}")

    lr           = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [32, 64])
    gamma_neg    = trial.suggest_float('asl_gamma_neg', 2.0, 6.0)
    gamma_pos    = trial.suggest_float('asl_gamma_pos', 0.0, 2.0)
    clip         = trial.suggest_float('asl_clip', 0.0, 0.1)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
    scheduler_t  = trial.suggest_categorical('scheduler', ['cosine', 'step', 'none'])
    rotation     = trial.suggest_int('rotation', 10, 20)
    translate    = trial.suggest_float('translate', 0.05, 0.15)

    print(f"  lr={lr:.2e} | batch={batch_size} | wd={weight_decay:.2e}")
    print(f"  ASL: γ_neg={gamma_neg:.2f}, γ_pos={gamma_pos:.2f}, clip={clip:.3f}")
    print(f"  scheduler={scheduler_t} | rot={rotation}° | translate={translate:.2f}")

    train_tf, val_tf = get_transforms(rotation=rotation, translate=translate)
    train_loader, val_loader = get_dataloaders(train_tf, val_tf, batch_size=batch_size)

    model     = build_model(N_CLASSES, pretrained=True).to(DEVICE)
    criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_t == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=lr * 0.01)
    elif scheduler_t == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    else:
        scheduler = None

    best_auc, no_improve = 0.0, 0
    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        print(f"    Epoch {epoch+1}/{cfg.n_epochs} — train...", end=' ', flush=True)
        run_one_epoch_train(model, train_loader, criterion, optimizer)
        print("val...", end=' ', flush=True)
        val_loss, auc, acc = run_one_epoch_val(model, val_loader, criterion, evaluator_val)
        print(f"AUC={auc:.4f} | loss={val_loss:.4f} | {time.time()-t0:.1f}s")

        if scheduler is not None:
            scheduler.step()

        trial.report(auc, epoch)
        if trial.should_prune():
            print(f"    → PRUNED en epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

        if auc > best_auc + 1e-4:
            best_auc, no_improve = auc, 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"    → Early stop interno en epoch {epoch+1}")
                break

    print(f"  Trial {trial.number + 1} completado — mejor AUC: {best_auc:.4f}")
    return best_auc


def run_hparam_search(cfg: OptunaConfig | None = None) -> dict:
    """Ejecuta la búsqueda de hiperparámetros y devuelve los mejores parámetros."""
    if cfg is None:
        cfg = OptunaConfig()

    evaluator_val = Evaluator(DATA_FLAG, split='val')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _objective(trial, evaluator_val, cfg),
        n_trials=cfg.n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_auc    = study.best_value

    print(f"\n{'='*50}")
    print(f"  BÚSQUEDA COMPLETADA")
    print(f"  Mejor AUC en val: {best_auc:.4f}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*50}")

    with open(OPTUNA_PARAMS_PATH, 'w') as f:
        json.dump({'best_auc': best_auc, 'best_params': best_params}, f, indent=2)
    print(f"Parámetros guardados en: {OPTUNA_PARAMS_PATH}")

    plot_optuna_search(study)

    return best_params
