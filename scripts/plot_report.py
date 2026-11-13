"""Generate auditable training figures from a saved evaluation report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    path = Path("models/metrics.json")
    report = json.loads(path.read_text())
    matrix = np.asarray(report["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Held-out test confusion matrix", xlabel="Predicted digit", ylabel="True digit", xticks=range(10), yticks=range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    recalls = report["test"]["per_class_recall"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), recalls)
    ax.set(title="Test recall by digit", xlabel="True digit", ylabel="Recall", xticks=range(10), ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_recall.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
