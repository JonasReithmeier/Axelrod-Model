import pandas as pd
from pathlib import Path
import yaml
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

print('Starting')
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
data_path = Path("data/task1")
master_file = data_path / (config['grid_visualization']['database_name'] + '.parquet')
master_file2 = data_path / (config['grid_visualization']['database_name'] + 'OLD.parquet')
old_df = pd.read_parquet(master_file)
old_df2 = pd.read_parquet(master_file2)
existing_data_map = {
            'width': [],
            'q': [],
            'F':[],
            'iterationNumber': [],
            'seed': [],
            's_max': [],
            's_mean': [],
            'is_frozen': [],
            'steps_to_freeze': []
}

existing_data_map2 = {
            'width': [],
            'q': [],
            'F':[],
            'iterationNumber': [],
            'seed': [],
            's_max': [],
            's_mean': [],
            'is_frozen': [],
            'steps_to_freeze': []
}

for row in old_df.itertuples(index=False):
    existing_data_map['width'].append(row.width)
    existing_data_map['q'].append(row.q)
    existing_data_map['F'].append(row.F)
    existing_data_map['iterationNumber'].append(row.iterationNumber)
    existing_data_map['seed'].append(row.seed)
    existing_data_map['s_max'].append(row.s_max)
    existing_data_map['s_mean'].append(row.s_mean)
    existing_data_map['is_frozen'].append(row.is_frozen)
    existing_data_map['steps_to_freeze'].append(row.steps_to_freeze)

for row in old_df2.itertuples(index=False):
    existing_data_map2['width'].append(row.width)
    existing_data_map2['q'].append(row.q)
    existing_data_map2['F'].append(row.F)
    existing_data_map2['iterationNumber'].append(row.iterationNumber)
    existing_data_map2['seed'].append(row.seed)
    existing_data_map2['s_max'].append(row.s_max)
    existing_data_map2['s_mean'].append(row.s_mean)
    existing_data_map2['is_frozen'].append(row.is_frozen)
    existing_data_map2['steps_to_freeze'].append(row.steps_to_freeze)

is_Equal = False
if compare(existing_data_map['width'],existing_data_map2['width']):
     

def compare(value1, value2):
    if len(value1) == len(value2):
        for i in len(value1):
            if(value1[i] != value2[i]): 
                return False
    else:
        return False 
    return True
