"""Giao thức đánh giá thống nhất cho mọi mô hình (phase 4)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import LABELS, RANDOM_STATE


def find_best_f1_threshold(y_true, proba) -> float:
    """Ngưỡng tối đa hóa F1 (chỉ được gọi trên tập VALIDATION)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def threshold_for_fpr(y_true, proba, max_fpr: float) -> float:
    """Ngưỡng nhỏ nhất (recall lớn nhất) vẫn giữ FPR ≤ max_fpr."""
    fpr, _, thr = roc_curve(y_true, proba)
    return float(thr[fpr <= max_fpr][-1])


def recall_at_fpr(y_true, proba, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.interp(max_fpr, fpr, tpr))


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, proba),
        'PR-AUC': average_precision_score(y_true, proba),
        'Recall@FPR≤1%': recall_at_fpr(y_true, proba, 0.01),
        'FPR': fp / (fp + tn), 'FNR': fn / (fn + tp),
        'Brier': brier_score_loss(y_true, np.clip(proba, 0, 1)),
        'Threshold': threshold, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 200, alpha: float = 0.05,
                 seed: int = RANDOM_STATE) -> dict:
    """Khoảng tin cậy bootstrap (percentile) cho F1, PR-AUC, ROC-AUC, Recall@FPR≤1%."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n = len(y_true)
    stats = {'F1-Score': [], 'PR-AUC': [], 'ROC-AUC': [], 'Recall@FPR≤1%': []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pp = y_true[idx], proba[idx]
        stats['F1-Score'].append(f1_score(yt, (pp >= threshold).astype(int)))
        stats['PR-AUC'].append(average_precision_score(yt, pp))
        stats['ROC-AUC'].append(roc_auc_score(yt, pp))
        stats['Recall@FPR≤1%'].append(recall_at_fpr(yt, pp, 0.01))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in stats.items()}


def plot_evaluation(name, y_true, proba, threshold, split='TEST'):
    """ROC + Precision–Recall + Confusion Matrix cho một mô hình."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_score(y_true, proba):.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    axes[0].legend(loc='lower right')

    prec, rec, _ = precision_recall_curve(y_true, proba)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    axes[1].plot(rec, prec, lw=2, c='#e34a33', label=f'AP = {average_precision_score(y_true, proba):.4f}')
    axes[1].axhline(y_true.mean(), ls='--', c='k', lw=1, label=f'Ngẫu nhiên = {y_true.mean():.3f}')
    axes[1].scatter(recall_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
                    s=80, c='k', zorder=5, label=f'Ngưỡng {threshold:.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–Recall Curve', ylim=(0, 1.02))
    axes[1].legend(loc='lower left')

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABELS, cmap='Blues',
                                            values_format=',', ax=axes[2], colorbar=False)
    axes[2].set_title(f'Confusion Matrix (ngưỡng = {threshold:.3f})')
    axes[2].grid(False)
    fig.suptitle(f'{name}: đánh giá trên tập {split}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
