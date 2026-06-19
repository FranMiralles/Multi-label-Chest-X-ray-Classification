"""Pipeline de calibración de umbrales y evaluación final en test."""

import json

import numpy as np
import torch
from tqdm import tqdm
from medmnist import INFO

from src.config.config import (
    DATA_FLAG, DEVICE, FINAL_MODEL_PATH, FINAL_METRICS_PATH,
)
from src.data.dataloader import get_val_loader, get_test_loader
from src.entity.schemas import TrainingConfig
from src.features.transforms import get_transforms
from src.models.architecture import build_model
from src.models.losses import AsymmetricLoss
from src.models.evaluator import calibrate_thresholds, compute_test_metrics
from src.utils.plotting import (
    plot_roc_curves, plot_confusion_and_metrics,
)

info      = INFO[DATA_FLAG]
N_CLASSES = len(info['label'])


def _inference(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_score, y_true) sobre un dataloader."""
    y_score, y_true = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Inferencia"):
            outputs = model(inputs.to(DEVICE))
            y_score.append(torch.sigmoid(outputs).cpu().numpy())
            y_true.append(targets.numpy())
    return np.vstack(y_score), np.vstack(y_true)


def run_evaluation(model=None, cfg: TrainingConfig | None = None) -> dict:
    """Calibra umbrales en val y evalúa en test. Devuelve el dict de métricas.

    Si model es None, carga desde FINAL_MODEL_PATH.
    Si cfg es None, usa ASL por defecto (sólo para el cálculo de test_loss).
    """
    if model is None:
        model = build_model(N_CLASSES, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Modelo cargado desde: {FINAL_MODEL_PATH}")

    _, val_tf = get_transforms()

    # ── Calibración de umbrales en validación ─────────────────────────────────
    print("\nCalibrando umbrales en validación...")
    val_loader  = get_val_loader(val_tf)
    y_score_val, y_true_val = _inference(model, val_loader)

    optimal_thresholds, threshold_strategy, results_val = calibrate_thresholds(
        y_score_val, y_true_val, N_CLASSES)

    print(f"\n{'='*68}")
    print(f"  COMPARACIÓN DE ESTRATEGIAS DE UMBRAL (validación)")
    print(f"{'='*68}")
    print(f"{'Estrategia':<30} {'F1-mac':>7} {'F1-mic':>7} {'FN':>7} {'FP':>7}")
    print("-" * 68)
    for name, r in results_val.items():
        print(f"{name:<30} {r['f1_mac']:>7.4f} {r['f1_mic']:>7.4f} "
              f"{r['fn']:>7d} {r['fp']:>7d}")
    print(f"{'='*68}")
    print(f"\n  → Estrategia elegida: {threshold_strategy}")

    # ── Evaluación en test ────────────────────────────────────────────────────
    print("\nEvaluando en test set...")
    test_loader = get_test_loader(val_tf)
    y_score_test, y_true_test = _inference(model, test_loader)
    y_pred_test = (y_score_test >= optimal_thresholds).astype(int)

    metrics = compute_test_metrics(y_true_test, y_score_test, y_pred_test, N_CLASSES)

    print(f"\n{'='*50}")
    print(f"   RESULTADOS FINALES EN TEST SET")
    print(f"   Modelo: DenseNet-121 + DA + ASL")
    print(f"   Umbral: {threshold_strategy}")
    print(f"{'='*50}")
    print(f"- AUC-ROC Macro        : {metrics['auc_macro']:.4f}")
    print(f"- Hamming Score        : {metrics['h_score']:.4f}")
    print(f"- Mean Balanced Acc    : {np.mean(metrics['balanced_accs']):.4f}")
    print(f"- F1-Score Macro       : {metrics['f1_macro']:.4f}")
    print(f"- F1-Score Micro       : {metrics['f1_micro']:.4f}")
    print(f"- Accuracy Global (EMR): {metrics['acc_global']:.4f}")
    print(f"- Falsos Negativos     : {metrics['fn_test']}")
    print(f"- Falsos Positivos     : {metrics['fp_test']}")
    print(f"{'='*50}")

    class_names = [info['label'][str(i)] for i in range(N_CLASSES)]
    print(f"\n{'Clase':<22} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} "
          f"{'FN':>6} {'FP':>6}")
    print("-" * 60)
    for i in range(N_CLASSES):
        print(f"{class_names[i][:22]:<22} "
              f"{metrics['roc_aucs'][i]:>6.3f} "
              f"{metrics['f1_per_class'][i]:>6.3f} "
              f"{metrics['sensitivity'][i]:>6.3f} "
              f"{metrics['specificity'][i]:>6.3f} "
              f"{metrics['fn_per_class'][i]:>6d} "
              f"{metrics['fp_per_class'][i]:>6d}")

    # ── Guardar métricas ──────────────────────────────────────────────────────
    final_metrics = {
        'auc_macro':         metrics['auc_macro'],
        'f1_macro':          metrics['f1_macro'],
        'f1_micro':          metrics['f1_micro'],
        'hamming_score':     metrics['h_score'],
        'accuracy_emr':      metrics['acc_global'],
        'mean_balanced_acc': float(np.mean(metrics['balanced_accs'])),
        'false_negatives':   metrics['fn_test'],
        'false_positives':   metrics['fp_test'],
        'threshold_strategy': threshold_strategy,
        'per_class': {
            class_names[i]: {
                'auc':         float(metrics['roc_aucs'][i]),
                'f1':          float(metrics['f1_per_class'][i]),
                'sensitivity': float(metrics['sensitivity'][i]),
                'specificity': float(metrics['specificity'][i]),
                'fn':          metrics['fn_per_class'][i],
                'fp':          metrics['fp_per_class'][i],
                'threshold':   float(optimal_thresholds[i]),
            } for i in range(N_CLASSES)
        },
    }
    with open(FINAL_METRICS_PATH, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nMétricas guardadas en: {FINAL_METRICS_PATH}")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    plot_roc_curves(y_true_test, y_score_test, metrics['roc_aucs'],
                    metrics['auc_macro'], info['label'])
    plot_confusion_and_metrics(y_true_test, y_pred_test,
                               metrics['roc_aucs'], info['label'])

    return final_metrics, optimal_thresholds, y_score_test, y_true_test, y_pred_test
