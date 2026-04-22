from pathlib import Path
import pandas as pd

csv_path = Path(__file__).resolve().parent.parent / "data" / "ppe_dataset" / "labels_classification.csv"

df = pd.read_csv(csv_path)

counts = df.groupby(["helmet", "vest", "glasses"]).size().reset_index(name="count")
counts = counts.sort_values(by="count", ascending=False)

print(counts)

print("\nHelmet:\n", df["helmet"].value_counts())
print("\nVest:\n", df["vest"].value_counts())
print("\nGlasses:\n", df["glasses"].value_counts())