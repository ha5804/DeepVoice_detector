import numpy as np
import matplotlib.pyplot as plt

def plot_score_distribution(scores, labels):
    normal_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    plt.figure(figsize=(8, 5))

    plt.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label="Normal (Human)",
        density=True
    )
    plt.hist(
        fake_scores,
        bins=50,
        alpha=0.6,
        label="DeepVoice (Fake)",
        density=True
    )

    plt.xlabel("Anomaly Score (Reconstruction Error)")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()
