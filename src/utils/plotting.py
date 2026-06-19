"""Funciones de visualización. Todos los archivos se guardan en reports/."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score

from src.config.config import REPORTS_DIR


def plot_optuna_search(study, save: bool = True) -> None:
    '''
    Plots the convergence of the Optuna hyperparameter search and the importance of each hyperparameter.
    Params:
        study: Optuna study object after optimization.
        save: Whether to save the plot to disk.
    '''
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    trial_values  = [t.value  for t in study.trials if t.value is not None]
    best_so_far   = np.maximum.accumulate(trial_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(trial_numbers, trial_values, 'o', alpha=0.5, label='Trial AUC')
    axes[0].plot(trial_numbers, best_so_far, '-', color='red', lw=2,
                 label='Best acumulated AUC')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('AUC Macro (val)')
    axes[0].set_title('Convergence of Hyperparameter Search')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        import optuna
        importances  = optuna.importance.get_param_importances(study)
        param_names  = list(importances.keys())
        param_values = list(importances.values())
        axes[1].barh(param_names[::-1], param_values[::-1], color='steelblue')
        axes[1].set_xlabel('Relative Importance')
        axes[1].set_title('Importance of Hyperparameters')
        axes[1].grid(True, alpha=0.3)
    except Exception:
        axes[1].text(0.5, 0.5, 'Importance not available\n(requires more trials)',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    if save:
        fpath = os.path.join(REPORTS_DIR, 'optuna_search.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()


def plot_training_curves(history: dict, save: bool = True) -> None:
    epochs_ran = range(1, len(history['auc']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs_ran, history['train_loss'], label='Train', marker='o', ms=3)
    axes[0].plot(epochs_ran, history['val_loss'],   label='Val',   marker='o', ms=3)
    axes[0].set_title('Pérdida (ASL)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_ran, history['auc'], color='tab:green', marker='o', ms=3)
    axes[1].set_title('AUC Macro (Val)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_ran, history['acc'], color='tab:orange', marker='o', ms=3)
    axes[2].set_title('Accuracy (Val)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Acc')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Curvas de Entrenamiento – DenseNet-121 (Final)', fontsize=13)
    plt.tight_layout()
    if save:
        fpath = os.path.join(REPORTS_DIR, 'training_curves.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()


def plot_roc_curves(y_true_test: np.ndarray, y_score_test: np.ndarray,
                    roc_aucs: list, auc_macro: float,
                    class_labels: dict, save: bool = True) -> None:
    n_classes = y_true_test.shape[1]
    colors    = plt.cm.tab20(np.linspace(0, 1, n_classes))

    plt.figure(figsize=(12, 9))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_test[:, i], y_score_test[:, i])
        plt.plot(fpr, tpr, color=colors[i],
                 label=f"{class_labels[str(i)]} (AUC={roc_aucs[i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curvas ROC – Test Set (AUC macro={auc_macro:.4f})\n'
              f'DenseNet-121 + DA + ASL + Optuna')
    plt.legend(loc='lower right', ncol=2, fontsize='x-small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fpath = os.path.join(REPORTS_DIR, 'roc_curves_test.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()


def plot_confusion_and_metrics(y_true_test: np.ndarray, y_pred_test: np.ndarray,
                               roc_aucs: list, class_labels: dict,
                               save: bool = True) -> None:
    n_classes    = y_true_test.shape[1]
    cm_global    = confusion_matrix(y_true_test.flatten(), y_pred_test.flatten())
    cm_norm      = cm_global.astype('float') / cm_global.sum(axis=1, keepdims=True)
    f1_per_class = f1_score(y_true_test, y_pred_test, average=None, zero_division=0)
    class_names  = [class_labels[str(i)] for i in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(cm_norm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Matriz de Confusión Global (Normalizada)', fontsize=12)
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Real')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Neg', 'Pos'])
    axes[0].set_yticklabels(['Neg', 'Pos'])
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, f'{cm_norm[row, col]:.3f}',
                         ha='center', va='center',
                         color='white' if cm_norm[row, col] > 0.5 else 'black',
                         fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[0])

    x_pos = np.arange(n_classes)
    width = 0.35
    axes[1].bar(x_pos - width / 2, roc_aucs,     width, label='AUC',
                color='steelblue', alpha=0.8)
    axes[1].bar(x_pos + width / 2, f1_per_class, width, label='F1',
                color='coral',     alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('AUC y F1 por Clase')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        fpath = os.path.join(REPORTS_DIR, 'confusion_and_metrics.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()
