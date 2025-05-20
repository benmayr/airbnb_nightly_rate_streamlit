"""
Main script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('../data/listings2.csv')

# Select specific columns
columns_to_keep = ['latitude', 'longitude', 'price', 'room_type', 
                  'minimum_nights', 'availability_365', 'neighbourhood_group']
df = df[columns_to_keep]

# Basic exploratory analysis
print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Distribution of room types
print("\nRoom Type Distribution:")
print(df['room_type'].value_counts())

# Distribution of neighbourhood groups
print("\nNeighbourhood Group Distribution:")
print(df['neighbourhood_group'].value_counts())

# Create some visualizations
plt.figure(figsize=(12, 6))

# Room type distribution
plt.subplot(1, 2, 1)
df['room_type'].value_counts().plot(kind='bar')
plt.title('Distribution of Room Types')
plt.xticks(rotation=45)

# Price distribution by room type
plt.subplot(1, 2, 2)
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../data/analysis_plots.png')
plt.close()

# Correlation analysis
numeric_columns = ['price', 'minimum_nights', 'availability_365']
correlation = df[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(correlation)

# Save the processed data
df.to_csv('../data/processed_listings.csv', index=False) 