import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("research_part1/processed", exist_ok=True)

def clean_dataset(input_path, output_path):

    print("Processing:", input_path)

    df = pd.read_csv(input_path)

    # remove byte strings
    df = df.applymap(lambda x: str(x).replace("b'", "").replace("'", "") if isinstance(x,str) else x)

    # --- IMPORTANT ---
    # last column is label → keep as categorical
    label_col = df.columns[-1]

    for col in df.columns:

        if col == label_col:
            # keep label categorical (string)
            df[col] = df[col].astype(str)

        else:
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    df.to_csv(output_path, index=False)
    print("Saved:", output_path)


clean_dataset("research_part1/datasets/electricity.csv",
              "research_part1/processed/electricity_clean.csv")

clean_dataset("research_part1/datasets/airlines_delay.csv",
              "research_part1/processed/airlines_clean.csv")

clean_dataset("research_part1/datasets/weather.csv",
              "research_part1/processed/weather_clean.csv")

print("\nDatasets cleaned correctly for classification!")
