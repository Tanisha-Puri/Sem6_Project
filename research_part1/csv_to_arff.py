import pandas as pd

def csv_to_arff(csv_path, arff_path, relation="stream"):

    df = pd.read_csv(csv_path)

    with open(arff_path, "w") as f:

        f.write(f"@RELATION {relation}\n\n")

        # attributes
        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        # class attribute (categorical)
        classes = df.iloc[:, -1].unique()
        classes = ",".join(map(str, classes))
        f.write(f"@ATTRIBUTE class {{{classes}}}\n\n")

        f.write("@DATA\n")

        for _, row in df.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")

    print("Saved:", arff_path)


csv_to_arff(
    "research_part1/processed/electricity_clean.csv",
    "research_part1/processed/electricity.arff",
    "electricity"
)

csv_to_arff(
    "research_part1/processed/airlines_clean.csv",
    "research_part1/processed/airlines.arff",
    "airlines"
)
