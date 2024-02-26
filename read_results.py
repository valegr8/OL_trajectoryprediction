import pandas as pd

# Specify the path to the Parquet file
parquet_file_path = 'submission.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Now you can work with the DataFrame as needed
# print(df.head())  # Display the first few rows of the DataFrame
print(df.columns )  # Display the first few rows of the DataFrame
grouped_df = df.groupby('scenario_id')
print(df['scenario_id'].drop_duplicates())

print(df[df['scenario_id'] == 'a0f31535-95c2-4d97-8a52-26e74f961cd5']['probability'])