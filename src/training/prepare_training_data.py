import pandas as pd

data = pd.read_csv("data/processed/clean_data.csv")

X = data[["SST", "SSS"]]

print("Training inputs:")
print(X)

print("\nInput shape:", X.shape)