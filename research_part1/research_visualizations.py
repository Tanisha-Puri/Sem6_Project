import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("research_part1/results.csv")

# Order streams
stream_order = ["abrupt_drift", "gradual_drift", "recurring_drift"]
df["stream"] = pd.Categorical(df["stream"], categories=stream_order, ordered=True)

# Output folder
os.makedirs("research_part1/figures", exist_ok=True)

sns.set(style="whitegrid", context="talk")

# -------------------------
# Process Each Stream
# -------------------------
for stream in df["stream"].unique():

    subset = df[df["stream"] == stream].copy()

    plt.figure(figsize=(14,7))

    # Smooth accuracy
    subset["smoothed"] = (
        subset.groupby("model")["accuracy"]
        .transform(lambda x: x.rolling(200, min_periods=1).mean())
    )

    # Plot smoothed accuracy
    sns.lineplot(
        data=subset,
        x="instance",
        y="smoothed",
        hue="model",
        linewidth=2
    )

    max_instance = subset["instance"].max()
    drift_point = max_instance // 2

    # -------------------------
    # Mark Drift (improved text placement)
    # -------------------------
    plt.axvline(x=drift_point, color="black", linestyle="--", linewidth=2)

    plt.text(
        drift_point + 50,
        0.9,   # place high to avoid overlap
        "Concept Drift",
        rotation=90,
        verticalalignment='top'
    )

    # -------------------------
    # Recovery Threshold
    # -------------------------
    pre_drift = subset[subset["instance"] < drift_point]
    pre_mean = pre_drift["accuracy"].mean()
    threshold = pre_mean * 0.9

    plt.axhline(y=threshold, linestyle=":", color="red")

    plt.text(
        subset["instance"].min() + 100,
        threshold + 0.01,
        "90% Recovery Threshold",
        color="red"
    )

    recovery_times = []

    # -------------------------
    # Mark recovery for each model (FIXED OVERLAP)
    # -------------------------
    offset_step = 0.02   # vertical spacing between labels
    label_index = 0

    for model in subset["model"].unique():

        model_data = subset[subset["model"] == model]
        after = model_data[model_data["instance"] > drift_point]
        recovered = after[after["smoothed"] >= threshold]

        if not recovered.empty:
            recovery_instance = recovered["instance"].iloc[0]
            recovery_time = recovery_instance - drift_point
            recovery_times.append([model, recovery_time])

            # plot marker
            plt.scatter(recovery_instance, threshold, s=70)

            # offset text vertically to prevent overlap
            plt.text(
                recovery_instance,
                threshold + (label_index * offset_step),
                model,
                fontsize=10
            )

            label_index += 1

    # -------------------------
    # Final formatting
    # -------------------------
    plt.title(f"Accuracy & Recovery Analysis - {stream.replace('_',' ').title()}")
    plt.xlabel("Instances")
    plt.ylabel("Smoothed Accuracy")
    plt.ylim(0,1)
    plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

    filename = f"research_part1/figures/{stream}_analysis.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved time-series plot: {filename}")

    # -------------------------
    # Recovery Time Bar Chart
    # -------------------------
    if recovery_times:
        recovery_df = pd.DataFrame(recovery_times, columns=["model", "recovery_time"])

        plt.figure(figsize=(8,5))
        sns.barplot(data=recovery_df, x="model", y="recovery_time")
        plt.title(f"Recovery Time Comparison – {stream}")
        plt.ylabel("Instances to Recover")
        plt.xlabel("Model")
        plt.tight_layout()

        filename = f"research_part1/figures/{stream}_recovery_time.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved recovery plot: {filename}")

    # -------------------------
    # Stability Boxplot
    # -------------------------
    plt.figure(figsize=(10,6))
    sns.boxplot(data=subset, x="model", y="accuracy")
    plt.title(f"Accuracy Distribution (Stability) – {stream}")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()

    filename = f"research_part1/figures/{stream}_stability.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved stability plot: {filename}")

print("\nAll research visualizations generated successfully!")