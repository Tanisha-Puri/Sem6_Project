import pandas as pd

df = pd.read_csv("research_part1/real_results.csv")

# get last instance accuracy per model per dataset
summary = df.groupby(["stream", "model"]).last().reset_index()

summary = summary[["stream", "model", "accuracy"]]

print("\nFinal Accuracy Summary:\n")
print(summary)

summary.to_csv("research_part1/final_accuracy_summary.csv", index=False)