
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv(r"runs\RSNA_1321_processed\history.csv")
df2 = pd.read_csv(r"runs\RSNA_1321_unprocessed\history.csv")

file1_column = "val_auc"
legend1 = "Validation AUC with Preprocessing"

file2_column = "val_auc"
legend2 = "Validation AUC without Preprocessing"

plt.figure(figsize=(8, 5))

plt.plot(
    df1["epoch"],
    df1[file1_column],
    marker="o",
    label=legend1
)

plt.plot(
    df2["epoch"],
    df2[file2_column],
    marker="s",
    label=legend2
)

plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.title("Validation AUC Comparison")
plt.ylim(0.6, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
