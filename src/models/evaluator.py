import numpy as np
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
        'Youden per class':                           thresholds_youden,
        f'F1-macro global (t={best_t_global:.2f})':   thresholds_global,
        'F1 per class':                               thresholds_f1_class,
        'Fixed 0.3':                                  np.full(n_classes, 0.3),
        'Fixed 0.4':                                  np.full(n_classes, 0.4),
        'Fixed 0.5':                                  np.full(n_classes, 0.5),
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
