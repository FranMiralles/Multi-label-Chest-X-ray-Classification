import json

import numpy as np
import torch
from tqdm import tqdm
from medmnist import INFO

from src.config.config import DATA_FLAG, DEVICE, FINAL_MODEL_PATH, FINAL_METRICS_PATH
from src.data.dataloader import get_val_loader, get_test_loader
from src.features.transforms import get_transforms
from src.models.architecture import build_densenet121 as build_model
from src.utils.plotting import plot_roc_curves, plot_confusion_and_metrics
from sklearn.metrics import (
    roc_curve, auc as metrics_auc,
    f1_score, accuracy_score,
    hamming_loss, balanced_accuracy_score,
    confusion_matrix,
)


def calibrate_thresholds(y_score_val: np.ndarray, y_true_val: np.ndarray,
                         n_classes: int) -> tuple:
    """Calculates optimal thresholds for multi-class classification.

    Returns:
        optimal_thresholds: optimal thresholds per class
        best_strategy: name of the best thresholding strategy
        results_val: dict with metrics for each strategy
    """
    # F1-macro global
    best_f1_global, best_t_global = 0.0, 0.5
    for t in np.linspace(0.05, 0.60, 60):
        f1 = f1_score(y_true_val, (y_score_val >= t).astype(int),
                      average='macro', zero_division=0)
        if f1 > best_f1_global:
            best_f1_global, best_t_global = f1, t
    thresholds_global = np.full(n_classes, best_t_global)

    # Youden per class
    thresholds_youden = np.zeros(n_classes)
    for i in range(n_classes):
        fpr, tpr, ths = roc_curve(y_true_val[:, i], y_score_val[:, i])
        thresholds_youden[i] = ths[np.argmax(tpr - fpr)]

    # F1 per class
    thresholds_f1_class = np.zeros(n_classes)
    for i in range(n_classes):
        best_f1_i, best_t_i = 0.0, 0.5
        for t in np.linspace(0.05, 0.60, 60):
            f1_i = f1_score(y_true_val[:, i],
                            (y_score_val[:, i] >= t).astype(int),
                            zero_division=0)
            if f1_i > best_f1_i:
                best_f1_i, best_t_i = f1_i, t
        thresholds_f1_class[i] = best_t_i

    strategies = {
        'Youden per class': thresholds_youden,
        f'F1-macro global (t={best_t_global:.2f})': thresholds_global,
        'F1 per class': thresholds_f1_class,
        'Fixed 0.3': np.full(n_classes, 0.3),
        'Fixed 0.4': np.full(n_classes, 0.4),
        'Fixed 0.5': np.full(n_classes, 0.5),
    }

    results_val = {}
    for name, thresholds in strategies.items():
        y_pred = (y_score_val >= thresholds).astype(int)
        results_val[name] = dict(
            f1_mac=f1_score(y_true_val, y_pred, average='macro',  zero_division=0),
            f1_mic=f1_score(y_true_val, y_pred, average='micro',  zero_division=0),
            fn=int(((y_true_val == 1) & (y_pred == 0)).sum()),
            fp=int(((y_true_val == 0) & (y_pred == 1)).sum()),
        )

    best_strategy = max(results_val, key=lambda k: results_val[k]['f1_mac'])
    return strategies[best_strategy], best_strategy, results_val


def compute_test_metrics(y_true_test: np.ndarray, y_score_test: np.ndarray,
                         y_pred_test: np.ndarray, n_classes: int) -> dict:
    """Calculates the complete set of metrics over the test set."""
    roc_aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_test[:, i], y_score_test[:, i])
        roc_aucs.append(metrics_auc(fpr, tpr))

    balanced_accs = [balanced_accuracy_score(y_true_test[:, i], y_pred_test[:, i])
                     for i in range(n_classes)]

    fn_per_class, fp_per_class, sensitivity, specificity = [], [], [], []
    for i in range(n_classes):
        TP = int(((y_true_test[:, i] == 1) & (y_pred_test[:, i] == 1)).sum())
        FN = int(((y_true_test[:, i] == 1) & (y_pred_test[:, i] == 0)).sum())
        FP = int(((y_true_test[:, i] == 0) & (y_pred_test[:, i] == 1)).sum())
        TN = int(((y_true_test[:, i] == 0) & (y_pred_test[:, i] == 0)).sum())
        fn_per_class.append(FN)
        fp_per_class.append(FP)
        sensitivity.append(TP / (TP + FN + 1e-8))
        specificity.append(TN / (TN + FP + 1e-8))

    return dict(
        roc_aucs=roc_aucs,
        auc_macro=float(np.mean(roc_aucs)),
        f1_macro=float(f1_score(y_true_test, y_pred_test, average='macro',  zero_division=0)),
        f1_micro=float(f1_score(y_true_test, y_pred_test, average='micro',  zero_division=0)),
        f1_per_class=f1_score(y_true_test, y_pred_test, average=None, zero_division=0),
        h_score=float(1 - hamming_loss(y_true_test, y_pred_test)),
        acc_global=float(accuracy_score(y_true_test, y_pred_test)),
        balanced_accs=balanced_accs,
        fn_test=int(((y_true_test == 1) & (y_pred_test == 0)).sum()),
        fp_test=int(((y_true_test == 0) & (y_pred_test == 1)).sum()),
        fn_per_class=fn_per_class,
        fp_per_class=fp_per_class,
        sensitivity=sensitivity,
        specificity=specificity,
    )


def _inference(model, loader) -> tuple[np.ndarray, np.ndarray]:
    y_score, y_true = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Inferencia"):
            outputs = model(inputs.to(DEVICE))
            y_score.append(torch.sigmoid(outputs).cpu().numpy())
            y_true.append(targets.numpy())
    return np.vstack(y_score), np.vstack(y_true)


def run_evaluation(model=None, cfg=None) -> tuple:
    """Calibra umbrales en val y evalúa en test. Devuelve (metrics, thresholds, scores, true, pred)."""
    info = INFO[DATA_FLAG]
    n_classes = len(info['label'])

    if model is None:
        model = build_model(n_classes, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Modelo cargado desde: {FINAL_MODEL_PATH}")

    _, val_tf = get_transforms()

    print("\nCalibrando umbrales en validación...")
    val_loader = get_val_loader(val_tf)
    y_score_val, y_true_val = _inference(model, val_loader)

    optimal_thresholds, threshold_strategy, results_val = calibrate_thresholds(
        y_score_val, y_true_val, n_classes)

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

    print("\nEvaluando en test set...")
    test_loader = get_test_loader(val_tf)
    y_score_test, y_true_test = _inference(model, test_loader)
    y_pred_test = (y_score_test >= optimal_thresholds).astype(int)

    metrics = compute_test_metrics(y_true_test, y_score_test, y_pred_test, n_classes)
    class_names = [info['label'][str(i)] for i in range(n_classes)]

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

    final_metrics = {
        'auc_macro':          metrics['auc_macro'],
        'f1_macro':           metrics['f1_macro'],
        'f1_micro':           metrics['f1_micro'],
        'hamming_score':      metrics['h_score'],
        'accuracy_emr':       metrics['acc_global'],
        'mean_balanced_acc':  float(np.mean(metrics['balanced_accs'])),
        'false_negatives':    metrics['fn_test'],
        'false_positives':    metrics['fp_test'],
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
            } for i in range(n_classes)
        },
    }
    with open(FINAL_METRICS_PATH, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nMétricas guardadas en: {FINAL_METRICS_PATH}")

    plot_roc_curves(y_true_test, y_score_test, metrics['roc_aucs'],
                    metrics['auc_macro'], info['label'])
    plot_confusion_and_metrics(y_true_test, y_pred_test,
                               metrics['roc_aucs'], info['label'])

    return final_metrics, optimal_thresholds, y_score_test, y_true_test, y_pred_test
