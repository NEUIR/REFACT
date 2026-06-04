import pandas as pd
import sys

try:
    df = pd.read_parquet("/user/xuxiaoyue/rldata/frames_fix/train.parquet")
    print("Columns:", df.columns)
    if 'data_source' in df.columns:
        print("Unique data_sources:", df['data_source'].unique())
    print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(f"Error reading frames_fix: {e}")

print("-" * 20)

try:
    df = pd.read_parquet("/user/xuxiaoyue/rldata/searchagent_mixed/train.parquet")
    print("Columns:", df.columns)
    if 'data_source' in df.columns:
        print("Unique data_sources:", df['data_source'].unique())
except Exception as e:
    print(f"Error reading searchagent_mixed: {e}")
