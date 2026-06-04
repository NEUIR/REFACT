import pyarrow.parquet as pq
import sys

file_paths = [
    "/user/xuxiaoyue/rldata/frames_fix/train.parquet",
    "/user/xuxiaoyue/rldata/searchagent_mixed/train.parquet"
]

for path in file_paths:
    print(f"Reading {path}...")
    try:
        table = pq.read_table(path)
        print("Columns:", table.column_names)
        if 'data_source' in table.column_names:
            unique_sources = table.column('data_source').unique().to_pylist()
            print("Unique data_sources:", unique_sources)
        else:
            print("No 'data_source' column found.")
            
        # Print first row sample of reward_model key if exists
        if 'reward_model' in table.column_names:
             print("First row reward_model:", table.column('reward_model')[0])
             
    except Exception as e:
        print(f"Error reading {path}: {e}")
    print("-" * 20)
