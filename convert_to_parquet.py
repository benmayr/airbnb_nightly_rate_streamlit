import pandas as pd
import os

# Create data/clean directory if it doesn't exist
os.makedirs("data/clean", exist_ok=True)

# Read CSV and save as parquet
df = pd.read_csv("data/clean/listings.csv")
df.to_parquet("data/clean/listings.parquet", index=False)
print("Converted listings.csv to listings.parquet") 