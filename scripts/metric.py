import numpy as np
import matplotlib.pyplot as plt

epochs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
losses = np.array([1, 0.03117187718550364, 0.029498561626921098, 0.022242434493576487, 0.017138139298185705, 0.013458665235278507, 0.017308281905328234, 0.012260314721303681, 0.01156983319669962, 0.009370721985275547, 0.0092326306233493])

auc = np.trapezoid(losses, epochs)

print(f"AUC = {auc:.6f}")

plt.plot(epochs, losses, marker="o")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()