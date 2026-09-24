import pandas as pd


# Load satellite-style data
data = pd.read_csv("data/raw/sample_data.csv")
print("Satellite-style data loaded successfully")
print("-----------------------------------------")
print()

print(data)

print()
print("Number of locations:", len(data))
print("Columns:", data.columns.tolist())