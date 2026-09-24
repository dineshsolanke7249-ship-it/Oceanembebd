import pandas as pd

data = pd.read_csv("data/raw/deep_ocean_training.csv")

print("Deep ocean training data:")
print(data)

print("\nColumns:")
print(data.columns.tolist())

print("\nData shape:")
print(data.shape)