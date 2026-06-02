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
old_df = pd.read_parquet(master_file)
#existing_data_map = {
#            (row.width, row.q, row.F, row.iterationNumber, row.is_frozen, row.steps_to_freeze) 
#            for row in old_df.itertuples(index=False)
#           
#}
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
########################################-----------UPDATE DATABASE-------------------######################################
#num = 0
#for val in existing_data_map:
#    if val['is_frozen'] == True:
#        print(f"Encountered false: {num}")
#        num += 1

###########################################################################################################################

#array_map = {
#    'width': [],
#    'q': [],
#    'F': [],
#    'iterationNumber':[],
#    'is_frozen': [],
#    'steps_to_freeze':[]
#}
#for row,value in existing_data_map.items():
#    array_map['width'] += [row[0]]
#    array_map['q'] += row[1]
#    array_map['F'] += row[2]
#    array_map['iterationNumber'] += row[3]
#    array_map['is_frozen'] += value[0]
#    array_map['steps_to_freeze'] += value[1]
    

#print(array_map)
#WRITE---------------#################################################################
#pq.write_table(pa.Table.from_pandas(pd.DataFrame(existing_data_map)), master_file)

#old_df = pd.read_parquet(master_file)
#filtered = old_df.to_table(
#    filter=(ds.field('is_frozen') == False)
#)

#df_filtered = filtered.to_pandas()
#print(f"Matching transactions: {len(df_filtered)}")

#print(df_filtered)