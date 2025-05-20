import pandas as pd

# Read both CSV and parquet
df_csv = pd.read_csv("data/clean/listings.csv")
df_parquet = pd.read_parquet("data/clean/listings.parquet")

print("CSV columns:", df_csv.columns.tolist())
print("\nParquet columns:", df_parquet.columns.tolist())
print("\nCSV head:\n", df_csv.head())
print("\nParquet head:\n", df_parquet.head()) 