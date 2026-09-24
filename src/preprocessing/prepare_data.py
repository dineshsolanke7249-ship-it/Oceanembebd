import pandas as pd
import os

input_file = "data/raw/sample_data.csv"
output_file = "data/processed/clean_data.csv"

data = pd.read_csv(input_file)

print("Original data:")
print(data)

data = data.dropna()

os.makedirs("data/processed", exist_ok=True)

data.to_csv(output_file, index=False)

print("\nPreprocessing completed!")
print("File saved at:", os.path.abspath(output_file))