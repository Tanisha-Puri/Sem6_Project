import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load real dataset results
df = pd.read_csv("research_part1/real_results.csv")

# Output folder
os.makedirs("research_part1/figures_real", exist_ok=True)

sns.set(style="whitegrid", context="talk")

# Plot per dataset
for stream in df["stream"].unique():

    subset = df[df["stream"] == stream]

    plt.figure(figsize=(12,6))

    sns.lineplot(
        data=subset,
        x="instance",
        y="accuracy",
        hue="model",
        linewidth=2
    )

    plt.title(f"Real Dataset: {stream.title()} Stream Performance")
    plt.xlabel("Number of Instances")
    plt.ylabel("Prequential Accuracy")

    plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

    filename = f"research_part1/figures_real/{stream}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", filename)

print("\nReal dataset plots generated successfully!")
