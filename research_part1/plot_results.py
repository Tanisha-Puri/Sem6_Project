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

    # Smooth accuracy (rolling mean)
    subset["smoothed_accuracy"] = (
        subset.groupby("model")["accuracy"]
        .transform(lambda x: x.rolling(200, min_periods=1).mean())
    )

    sns.lineplot(
        data=subset,
        x="instance",
        y="smoothed_accuracy",
        hue="model",
        linewidth=2
    )

    # Mark drift point (synthetic streams)
    max_instance = subset["instance"].max()
    drift_point = max_instance // 2

    plt.axvline(x=drift_point, color="black", linestyle="--", linewidth=2)
    plt.text(drift_point+50, 0.5, "Drift Occurs", rotation=90)

    # Optional: shade recovery region
    plt.axvspan(drift_point, drift_point+1000, alpha=0.1)

    plt.title(f"Prequential Accuracy under {stream.replace('_',' ').title()}")
    plt.xlabel("Number of Instances Seen")
    plt.ylabel("Smoothed Prequential Accuracy")
    plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.ylim(0,1)
    plt.tight_layout()

    filename = f"research_part1/figures/{stream}_enhanced.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nAll plots generated successfully!")
