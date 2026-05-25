import pandas as pd
from pathlib import Path


# Path to your master database
db_path = Path("data/schelling/schelling_master_results.parquet")

# Load data
df = pd.read_parquet(db_path)

total_entries = len(df)
non_frozen_df = df[df['is_frozen'] == False]
frozen_df = df[df['is_frozen'] == True]

num_non_frozen = len(non_frozen_df)
num_frozen = len(frozen_df)

df_specific = df.loc[
        (df['width'] == 20) & 
        (df['F'] == 3) & 
        (df['h'] == 0.05) & 
        (df['T'].isin([0.2, 0.8])) & 
        (df['q'].isin([10, 20, 40, 80, 120, 160, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 8000]))]

print(df_specific['steps_to_freeze'].agg(['mean', 'std']))
print(df_specific)

# threshold for check 10K; for mobility observe 10K (MCS) : mean    1.769315e+07    std     2.261264e+07


