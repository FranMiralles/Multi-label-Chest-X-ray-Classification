"""Pipeline de entrenamiento final con los mejores hiperparámetros."""

import json
import os

import torch
import torch.optim as optim
from medmnist import INFO, Evaluator

from src.config.config import DATA_FLAG, DEVICE, FINAL_MODEL_PATH, OPTUNA_PARAMS_PATH
from src.data.dataloader import get_dataloaders
from src.entity.schemas import TrainingConfig
from src.features.transforms import get_transforms
from src.models.architecture import build_model
from src.models.losses import AsymmetricLoss
from src.models.trainer import EarlyStopping, run_one_epoch_train, run_one_epoch_val
from src.utils.plotting import plot_training_curves

info = INFO[DATA_FLAG]
N_CLASSES = len(info['label'])


def run_training(best_params: dict | None = None,
                 train_cfg: TrainingConfig | None = None) -> tuple:
    """Entrena el modelo final y devuelve (model, history, cfg).

    Args:
        best_params: hiperparámetros de Optuna (lr, batch_size, asl_*, scheduler,
                     rotation, translate). Si None, los carga desde disco.
        train_cfg: controla n_epochs y patience. Si None, usa TrainingConfig()
                   con sus valores por defecto definidos en entity/schemas.py.
    """
    if best_params is None:
        with open(OPTUNA_PARAMS_PATH) as f:
            saved = json.load(f)
        best_params = saved['best_params']
        print(f"Usando parámetros guardados (AUC val = {saved['best_auc']:.4f})")

    # Hiperparámetros de Optuna + defaults de n_epochs/patience del schema
    cfg = TrainingConfig.from_dict(best_params)
    if train_cfg is not None:
        cfg.n_epochs = train_cfg.n_epochs
        cfg.patience = train_cfg.patience

    print(f"\n{'='*50}")
    print(f"  CONFIGURACIÓN ENTRENAMIENTO FINAL")
    print(f"  LR: {cfg.lr:.2e} | Batch: {cfg.batch_size} | WD: {cfg.weight_decay:.2e}")
    print(f"  ASL: γ_neg={cfg.asl_gamma_neg:.2f}, γ_pos={cfg.asl_gamma_pos:.2f}, "
          f"clip={cfg.asl_clip:.3f}")
    print(f"  Scheduler: {cfg.scheduler} | DA: rot={cfg.rotation}°, "
          f"translate={cfg.translate:.2f}")
    print(f"  Epochs máx: {cfg.n_epochs} | Patience: {cfg.patience}")
    print(f"{'='*50}")

    train_tf, val_tf = get_transforms(rotation=cfg.rotation, translate=cfg.translate)
    train_loader, val_loader = get_dataloaders(train_tf, val_tf, batch_size=cfg.batch_size)

    if os.path.exists(FINAL_MODEL_PATH):
        print(f"\nModelo previo encontrado. Cargando pesos...")
        model = build_model(N_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    else:
        print("\nInicializando con pesos ImageNet...")
        model = build_model(N_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    criterion = AsymmetricLoss(gamma_neg=cfg.asl_gamma_neg,
                               gamma_pos=cfg.asl_gamma_pos, clip=cfg.asl_clip)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01)
    elif cfg.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True,
                                   path=FINAL_MODEL_PATH)
    evaluator_val = Evaluator(DATA_FLAG, split='val')
    history = {'train_loss': [], 'val_loss': [], 'auc': [], 'acc': []}

    print("\nIniciando entrenamiento final...\n")
    for epoch in range(cfg.n_epochs):
        train_loss = run_one_epoch_train(model, train_loader, criterion, optimizer)
        val_loss, auc, acc = run_one_epoch_val(model, val_loader, criterion, evaluator_val)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['auc'].append(auc)
        history['acc'].append(acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>3}/{cfg.n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"AUC: {auc:.4f} | Acc: {acc:.4f} | LR: {lr_now:.2e}")

        early_stopping(auc, model)
        if early_stopping.early_stop:
            print(f"\n  Early stopping en epoch {epoch+1}.")
            break

    print(f"\n  Mejor AUC: {early_stopping.auc_max:.4f}")
    print(f"  Modelo guardado en: {FINAL_MODEL_PATH}")

    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    model.eval()

    plot_training_curves(history)

    return model, history, cfg
