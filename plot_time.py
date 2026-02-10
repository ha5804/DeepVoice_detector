import numpy as np
import matplotlib.pyplot as plt

ae_train = np.load("ae_train_times.npy")
clf_train = np.load("clf_train_times.npy")

ae_test = np.load("ae_test_time.npy")[0]
clf_test = np.load("clf_test_time.npy")[0]

# -------- Train time plot --------
plt.figure(figsize=(8, 4))
plt.plot(ae_train, label="LSTM-AE Train Time")
plt.plot(clf_train, label="Classifier Train Time")
plt.xlabel("Epoch")
plt.ylabel("Time (sec)")
plt.title("Training Time per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# -------- Test time bar plot --------
plt.figure(figsize=(6, 4))
plt.bar(
    ["LSTM-AE", "Classifier"],
    [ae_test, clf_test]
)
plt.ylabel("Total Test Time (sec)")
plt.title("Total Inference Time Comparison")
plt.grid(True)
plt.show()
