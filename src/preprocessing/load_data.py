import pandas as pd

file_path = "data/raw/sample_data.csv"

data = pd.read_csv(file_path)

print("Data loaded successfully!")
print("Shape:", data.shape)
print(data.head())