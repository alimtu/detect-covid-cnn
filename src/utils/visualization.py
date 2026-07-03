"""Plotting helpers for training curves and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_history(
    history: Mapping[str, Sequence[float]],
    output_dir: str | Path,
) -> None:
    """Save accuracy and loss curves from a training history.

    Args:
        history: Mapping with keys ``train_loss``, ``val_loss``,
            ``train_acc`` and ``val_acc``.
        output_dir: Directory where ``accuracy.png`` and ``loss.png`` are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=150)
    plt.close()

    # Accuracy curve.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png", dpi=150)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> dict[str, float]:
    """Save one-vs-rest ROC curves and return per-class + macro AUC.

    Args:
        y_true: Integer labels of shape ``(n,)``.
        y_score: Predicted probabilities of shape ``(n, n_classes)``.
        class_names: Ordered class labels.
        output_path: File path for the saved figure.

    Returns:
        Mapping of ``<class> -> AUC`` plus a ``macro`` entry.
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    aucs: dict[str, float] = {}
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        area = auc(fpr, tpr)
        aucs[name] = float(area)
        plt.plot(fpr, tpr, label=f"{name} (AUC={area:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    aucs["macro"] = float(np.mean(list(aucs.values()))) if aucs else 0.0
    return aucs


def plot_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> dict[str, float]:
    """Save one-vs-rest precision-recall curves and return per-class AP.

    Args:
        y_true: Integer labels of shape ``(n,)``.
        y_score: Predicted probabilities of shape ``(n, n_classes)``.
        class_names: Ordered class labels.
        output_path: File path for the saved figure.

    Returns:
        Mapping of ``<class> -> average precision`` plus a ``macro`` entry.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    aps: dict[str, float] = {}
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        aps[name] = float(ap)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (one-vs-rest)")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    aps["macro"] = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        matrix: Square confusion matrix of shape ``(n_classes, n_classes)``.
        class_names: Ordered class labels.
        output_path: File path for the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
