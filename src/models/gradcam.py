"""Grad-CAM++ explainability logic for DenseNet-121 on ChestMNIST."""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.config.config import DATA_FLAG, DEVICE, DOWNLOAD, REPORTS_DIR
from src.features.transforms import IMAGENET_MEAN, IMAGENET_STD

info = INFO[DATA_FLAG]
N_CLASSES = len(info['label'])


def _denormalize(tensor_chw: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = tensor_chw.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1).astype(np.float32)


def _label_str(label_vec: np.ndarray, class_names: list) -> list:
    names = [class_names[i] for i, v in enumerate(label_vec) if v == 1]
    return names if names else ['ninguna']


# ── Funciones de búsqueda de muestras ─────────────────────────────────────────

def find_single_class(all_true, all_probs, all_pred, cls_idx,
                      correct=True, n=4, min_prob=0.4):
    found = []
    for i in range(len(all_true)):
        only_one   = (all_true[i].sum() == 1) and (all_true[i, cls_idx] == 1)
        is_correct = (all_true[i, cls_idx] == all_pred[i, cls_idx])
        high_prob  = all_probs[i, cls_idx] >= min_prob
        if only_one and (is_correct == correct) and high_prob:
            found.append(i)
        if len(found) == n:
            break
    return found


def find_any_class(all_true, all_pred, cls_idx,
                   predicted_positive=True, correct=True, n=4):
    found = []
    for i in range(len(all_true)):
        has_class  = all_true[i, cls_idx] == 1
        pred_class = all_pred[i, cls_idx] == 1
        is_correct = (all_true[i, cls_idx] == all_pred[i, cls_idx])
        if has_class and (pred_class == predicted_positive) and (is_correct == correct):
            found.append(i)
        if len(found) == n:
            break
    return found


def find_false_positives(all_true, all_pred, cls_idx, n=4):
    return [i for i in range(len(all_true))
            if all_true[i, cls_idx] == 0 and all_pred[i, cls_idx] == 1][:n]


def find_false_negatives(all_true, all_pred, cls_idx, n=4):
    return [i for i in range(len(all_true))
            if all_true[i, cls_idx] == 1 and all_pred[i, cls_idx] == 0][:n]


def find_multilabel(all_true, all_pred, min_labels=3, correct=True, n=4):
    found = []
    for i in range(len(all_true)):
        many       = all_true[i].sum() >= min_labels
        is_correct = np.array_equal(all_true[i], all_pred[i])
        if many and (is_correct == correct):
            found.append(i)
        if len(found) == n:
            break
    return found


def find_high_confidence(all_true, all_probs, all_pred, cls_idx,
                         correct=True, n=4, min_prob=0.65):
    found = []
    for i in range(len(all_true)):
        has_class  = all_true[i, cls_idx] == 1
        high       = all_probs[i, cls_idx] >= min_prob
        is_correct = (all_true[i, cls_idx] == all_pred[i, cls_idx])
        if has_class and high and (is_correct == correct):
            found.append(i)
        if len(found) == n:
            break
    return found


# ── Plot principal ─────────────────────────────────────────────────────────────

def _plot_cam_panel(model, cam_extractor, test_ds, all_probs, all_pred, all_true,
                    indices_and_meta: list, class_names: list,
                    title: str, cols: int = 4) -> None:
    n = len(indices_and_meta)
    if n == 0:
        print(f"  [AVISO] Sin muestras para: {title}")
        return

    rows      = (n + cols - 1) // cols
    n_rows_ax = rows * 2
    fig, axes = plt.subplots(n_rows_ax, cols,
                             figsize=(cols * 3.8, n_rows_ax * 3.5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_rows_ax == 1 else axes.reshape(-1, 1)

    def get_ax(row, col):
        return axes[min(row, axes.shape[0]-1), min(col, axes.shape[1]-1)]

    class _MultiLabelTarget:
        def __init__(self, idx):
            self.idx = idx
        def __call__(self, output):
            if output.dim() == 1:
                return output[self.idx].unsqueeze(0)
            return output[:, self.idx]

    for k, (idx, cls_idx, desc) in enumerate(indices_and_meta):
        row_img = (k // cols) * 2
        row_cam = row_img + 1
        col     =  k % cols

        try:
            x_cam, label = test_ds[idx]
            true     = np.array(label).flatten().astype(int)
            inp      = x_cam.unsqueeze(0).to(DEVICE)
            probs    = all_probs[idx]
            pred     = all_pred[idx]
            img_vis  = _denormalize(x_cam)

            with torch.enable_grad():
                model.train()
                cam_map = cam_extractor(inp, targets=[_MultiLabelTarget(cls_idx)])[0]
                model.eval()
            overlay = show_cam_on_image(img_vis, cam_map, use_rgb=True)
        except Exception as e:
            print(f"  [AVISO] Error en muestra {idx}: {e}")
            get_ax(row_img, col).axis('off')
            get_ax(row_cam, col).axis('off')
            continue

        true_names = _label_str(true, class_names)
        pred_names = _label_str(pred, class_names)
        correct    = np.array_equal(true, pred)
        color      = 'green' if correct else 'red'
        cls_name   = class_names[cls_idx]

        ax_img = get_ax(row_img, col)
        ax_img.imshow(img_vis[:, :, 0], cmap='gray')
        ax_img.set_title(
            f"#{idx} | {desc}\n"
            f"CAM: {cls_name} (p={probs[cls_idx]:.2f})\n"
            f"Real: {', '.join(true_names[:3])}\n"
            f"Pred: {', '.join(pred_names[:3])}",
            fontsize=6.5, color=color, pad=2)
        ax_img.axis('off')
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        ax_cam = get_ax(row_cam, col)
        ax_cam.imshow(overlay)
        ax_cam.set_title(f"Grad-CAM++ → {cls_name}", fontsize=6.5, pad=2)
        ax_cam.axis('off')

    for k in range(n, rows * cols):
        row_img = (k // cols) * 2
        get_ax(row_img, k % cols).axis('off')
        get_ax(row_img + 1, k % cols).axis('off')

    legend = [mpatches.Patch(color='green', label='Predicción correcta'),
              mpatches.Patch(color='red',   label='Predicción incorrecta')]
    fig.legend(handles=legend, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()

    fname = f"gradcam_{title[:35].replace(' ', '_').replace('—', '')}.png"
    fpath = os.path.join(REPORTS_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {fpath}")


def run_gradcam(model, optimal_thresholds: np.ndarray) -> None:
    """Genera los 7 paneles de Grad-CAM++ y los guarda en reports/."""
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_ds = medmnist.ChestMNIST(split='test', transform=val_tf,
                                  download=DOWNLOAD, size=224)

    class_names  = [info['label'][str(i)] for i in range(N_CLASSES)]
    EASY_CLASSES = [class_names.index('emphysema'), class_names.index('hernia')]
    HARD_CLASSES = [class_names.index('infiltration'), class_names.index('pneumonia')]

    # Inferencia completa sobre test
    print("Calculando predicciones sobre test set completo...")
    all_probs, all_true = [], []
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            all_probs.append(p)
            all_true.append(np.array(yb).reshape(-1, N_CLASSES))
    all_probs = np.vstack(all_probs)
    all_true  = np.vstack(all_true).astype(int)
    all_pred  = (all_probs >= optimal_thresholds).astype(int)
    print(f"  {len(all_true)} muestras en test. Construyendo paneles...\n")

    # Configurar Grad-CAM++
    cam_extractor = GradCAMPlusPlus(model=model,
                                    target_layers=[model.features.denseblock4])

    def panel(indices_meta, title, cols=4):
        _plot_cam_panel(model, cam_extractor, test_ds,
                        all_probs, all_pred, all_true,
                        indices_meta, class_names, title, cols)

    # Panel 1 — Clases fáciles correctas
    print("Panel 1: Clases fáciles (emphysema, hernia) — correctas")
    easy_indices = []
    for cls in EASY_CLASSES:
        idxs = find_single_class(all_true, all_probs, all_pred, cls, correct=True, n=4)
        if len(idxs) < 4:
            idxs += find_any_class(all_true, all_pred, cls, correct=True,
                                   n=4 - len(idxs))
        easy_indices += [(i, cls, 'fácil·correcto') for i in idxs]
    panel(easy_indices, "Panel 1 — Clases fáciles (correctas)")

    # Panel 2 — Clases difíciles correctas
    print("Panel 2: Clases difíciles (infiltration, pneumonia) — correctas")
    hard_indices = []
    for cls in HARD_CLASSES:
        idxs = find_any_class(all_true, all_pred, cls, correct=True, n=4)
        hard_indices += [(i, cls, 'difícil·correcto') for i in idxs]
    panel(hard_indices, "Panel 2 — Clases difíciles (correctas)")

    # Panel 3 — Falsos positivos
    print("Panel 3: Falsos positivos")
    fp_indices = []
    for cls in EASY_CLASSES + HARD_CLASSES:
        idxs = find_false_positives(all_true, all_pred, cls, n=2)
        fp_indices += [(i, cls, 'FALSO POSITIVO') for i in idxs]
    panel(fp_indices, "Panel 3 — Falsos Positivos")

    # Panel 4 — Falsos negativos
    print("Panel 4: Falsos negativos")
    fn_indices = []
    for cls in EASY_CLASSES + HARD_CLASSES:
        idxs = find_false_negatives(all_true, all_pred, cls, n=2)
        fn_indices += [(i, cls, 'FALSO NEGATIVO') for i in idxs]
    panel(fn_indices, "Panel 4 — Falsos Negativos")

    # Panel 5 — Multi-label
    print("Panel 5: Casos multi-label (>=3 patologías)")
    ml_idxs  = find_multilabel(all_true, all_pred, min_labels=3, correct=True,  n=4)
    ml_idxs += find_multilabel(all_true, all_pred, min_labels=3, correct=False, n=4)
    ml_indices = []
    for i in ml_idxs[:8]:
        active = [c for c in range(N_CLASSES) if all_true[i, c] == 1]
        if not active:
            continue
        cls_chosen = max(active, key=lambda c: all_probs[i, c])
        label = 'multi·OK' if np.array_equal(all_true[i], all_pred[i]) \
                else 'multi·ERROR'
        ml_indices.append((i, cls_chosen, label))
    panel(ml_indices, "Panel 5 — Casos Multi-label")

    # Panel 6 — Alta confianza
    print("Panel 6: Alta confianza — correctos vs incorrectos")
    hc_indices = []
    for cls in EASY_CLASSES + HARD_CLASSES:
        for corr, label in [(True, 'conf-alta·OK'), (False, 'conf-alta·FALLO')]:
            idxs = find_high_confidence(all_true, all_probs, all_pred, cls,
                                        correct=corr, n=2, min_prob=0.65)
            hc_indices += [(i, cls, label) for i in idxs]
    panel(hc_indices, "Panel 6 — Alta Confianza OK vs Fallo")

    # Panel 7 — Una muestra por clase
    print("Panel 7: Una muestra correcta por cada una de las 14 clases")
    all_cls_indices = []
    for cls in range(N_CLASSES):
        idxs = find_any_class(all_true, all_pred, cls, correct=True, n=1)
        if not idxs:
            idxs = find_any_class(all_true, all_pred, cls, correct=False, n=1)
        if idxs:
            all_cls_indices.append((idxs[0], cls, class_names[cls]))
    panel(all_cls_indices, "Panel 7 — Una muestra por clase", cols=4)

    print("\n" + "="*55)
    print("  GRAD-CAM++ COMPLETADO — 7 paneles generados")
    print(f"  Guardados en: {REPORTS_DIR}/")
    print("="*55)
