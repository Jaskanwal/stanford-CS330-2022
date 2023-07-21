import pandas as pd
import matplotlib.pyplot as plt

# Results after running q3.sh
data = pd.DataFrame(
    [
        ["protonet", 1, 0.987, 0.002],
        ["protonet", 2, 0.994, 0.001],
        ["protonet", 4, 0.996, 0.001],
        ["protonet", 6, 0.997, 0.001],
        ["protonet", 8, 0.997, 0.001],
        ["protonet", 10, 0.996, 0.001],
        ["maml", 1, 0.978, 0.003],
        ["maml", 2, 0.989, 0.002],
        ["maml", 4, 0.991, 0.002],
        ["maml", 6, 0.993, 0.001],
        ["maml", 8, 0.994, 0.002],
        ["maml", 10, 0.993, 0.002],
    ],
    columns=["model", "K", "mean", "95_ci"]
)
# Calculate confidence intervals
data["lower_ci"] = data["mean"] - data["95_ci"]
data["upper_ci"] = data["mean"] + data["95_ci"]

# Make the plot
fig, ax = plt.subplots(1,1, figsize=(5, 4), facecolor="white")

for model, df in data.groupby("model"):
    ax.plot(df["K"].values, df["mean"].values, label=model)
    ax.fill_between(df["K"].values, df["lower_ci"].values, df["upper_ci"].values, alpha=0.2)

ax.grid(True)
ax.legend()
ax.set_xlabel("Number of Support Examples, K")
ax.set_ylabel("Query Accuracy")

plt.tight_layout()
plt.show()
