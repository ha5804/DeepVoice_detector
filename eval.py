from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(scores, labels):
    """
    scores: anomaly scores (higher = more anomalous)
    labels: 0 (normal), 1 (fake)
    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (DeepVoice Anomaly Detection)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc
