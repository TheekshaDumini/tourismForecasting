import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from typing import Dict, List


def get_confusion_matrix(model, ds, n_bins=5):
    bin_width = 1 / n_bins
    bins = np.arange(0, 1 + bin_width, bin_width)
    actls = []
    preds = []

    ds.set_country(None)
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            pred = model(x).squeeze()
            actl_bins = np.digitize(y, bins).tolist()
            pred_bins = np.digitize(pred, bins).tolist()
            actls.extend(actl_bins)
            preds.extend(pred_bins)

    labels = np.arange(1, len(bins))
    confusion_matrix = metrics.confusion_matrix(preds, actls, labels=labels)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=labels
    )

    return cm_display


def get_accuracy(pred, y, scale, count_thresh=5000):
    pred, y = pred.squeeze(), y.squeeze()
    accurate = np.abs(pred - y) * scale <= count_thresh
    return accurate


def plot_accuracy(history: Dict[str, List[float]], path: str) -> None:
    plt.plot(history["train_accuracy"], label="Train set")
    lines = plt.plot(history["test_accuracy"], label="Test set")
    min_test_loss = min(history["test_rmse"])
    min_test_loss_id = history["test_rmse"].index(min_test_loss)
    accepted_accuracy = history["test_accuracy"][min_test_loss_id]
    plt.vlines(min_test_loss_id, 0, accepted_accuracy, "black", "dashed")
    plt.hlines(accepted_accuracy, 0, min_test_loss_id, "black", "dashed")
    ax = lines[0].axes
    ax.set_xticks(list(ax.get_xticks()) + [min_test_loss_id])
    ax.set_yticks(list(ax.get_yticks()) + [accepted_accuracy])
    plt.xlim(0, len(history["test_accuracy"]))
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.suptitle("Variation of Accuracy over epochs")
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()
