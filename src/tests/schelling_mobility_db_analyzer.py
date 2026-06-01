import pandas as pd
from pathlib import Path
import numpy as np


# Path to your master database
db_path = Path("data/schelling/mobility_trajectories.parquet")

# Load data
df = pd.read_parquet(db_path)

total_entries = len(df)
print(total_entries)
print(df.columns)


"""
a = np.array([0,1,2,3,4,5,6,7,8,9])
print(a[1:])
print(a[:-1])
print(a[-2:])
print(a[:2])
print(a[-1])
"""