import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("research_part1/results.csv")

# ensure order
stream_order = ["abrupt_drift", "gradual_drift", "recurring_drift"]
df["stream"] = pd.Categorical(df["stream"], categories=stream_order, ordered=True)

# create output folder
os.makedirs("research_part1/figures", exist_ok=True)

# nicer style
sns.set(style="whitegrid", context="talk")

# -------------------------
# Plot each stream
# -------------------------
for stream in df["stream"].unique():

    subset = df[df["stream"] == stream].copy()

    plt.figure(figsize=(12,6))

    sns.lineplot(
        data=subset,
        x="instance",
        y="accuracy",
        hue="model",
        linewidth=2
    )

    plt.title(f"Prequential Accuracy under {stream.replace('_',' ').title()}")
    plt.xlabel("Number of Instances Seen")
    plt.ylabel("Prequential Accuracy")
    plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

    filename = f"research_part1/figures/{stream}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nAll plots generated successfully!")
