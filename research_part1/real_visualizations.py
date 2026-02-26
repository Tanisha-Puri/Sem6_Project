import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Load real dataset results
# -------------------------
df = pd.read_csv("research_part1/real_results.csv")

# Create output folder
os.makedirs("research_part1/figures_real", exist_ok=True)

sns.set(style="whitegrid", context="talk")

# ==========================================================
# 1️⃣ Smoothed Accuracy Curves (Per Dataset)
# ==========================================================
for stream in df["stream"].unique():

    subset = df[df["stream"] == stream].copy()

    plt.figure(figsize=(14,7))

    # Smooth accuracy
    subset["smoothed"] = (
        subset.groupby("model")["accuracy"]
        .transform(lambda x: x.rolling(300, min_periods=1).mean())
    )

    sns.lineplot(
        data=subset,
        x="instance",
        y="smoothed",
        hue="model",
        linewidth=2.5
    )

    plt.title(f"Real Dataset: {stream.title()} Stream Performance")
    plt.xlabel("Number of Instances")
    plt.ylabel("Smoothed Prequential Accuracy")
    plt.ylim(0,1)

    plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

    filename = f"research_part1/figures_real/{stream}_curve.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", filename)


# ==========================================================
# 2️⃣ Final Accuracy Comparison (Bar Chart)
# ==========================================================
final_acc = (
    df.groupby(["stream", "model"])["accuracy"]
    .last()
    .reset_index()
)

plt.figure(figsize=(12,6))

sns.barplot(
    data=final_acc,
    x="model",
    y="accuracy",
    hue="stream"
)

plt.title("Final Accuracy Comparison (Real Datasets)")
plt.ylabel("Final Prequential Accuracy")
plt.xlabel("Model")
plt.ylim(0,1)

plt.legend(title="Dataset", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()

filename = "research_part1/figures_real/final_accuracy_comparison.png"
plt.savefig(filename, dpi=300)
plt.close()

print("Saved:", filename)


# ==========================================================
# 3️⃣ Stability Comparison (Boxplot)
# ==========================================================
plt.figure(figsize=(12,6))

sns.boxplot(
    data=df,
    x="model",
    y="accuracy",
    hue="stream"
)

plt.title("Accuracy Distribution (Stability) – Real Datasets")
plt.ylabel("Prequential Accuracy")
plt.xlabel("Model")
plt.ylim(0,1)

plt.legend(title="Dataset", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()

filename = "research_part1/figures_real/stability_comparison.png"
plt.savefig(filename, dpi=300)
plt.close()

print("Saved:", filename)

print("\nAll real dataset visualizations generated successfully!")