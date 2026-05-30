import pandas as pd
from pathlib import Path
import yaml
import pyarrow.dataset as ds

print('Starting')
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
data_path = Path("data/task1")
master_file = data_path / (config['grid_visualization']['database_name'] + '.parquet')
old_df = pd.read_parquet(master_file)
existing_data_map = {
            (row.width, row.q, row.F, row.iterationNumber): (row.is_frozen, row.steps_to_freeze) 
            for row in old_df.itertuples(index=False)
}

########################################-----------UPDATE DATABASE-------------------######################################


###########################################################################################################################

existing_data_map.to_parquet(master_file, index=False)

#old_df = pd.read_parquet(master_file)
#filtered = old_df.to_table(
#    filter=(ds.field('is_frozen') == False)
#)

#df_filtered = filtered.to_pandas()
#print(f"Matching transactions: {len(df_filtered)}")

#print(df_filtered)