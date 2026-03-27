"""Read a parquet file and print column names and shapes."""

import sys
import pandas as pd

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/t-qimhuang/disk2/yanzhe_data/grid5/5grids/data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(path)
    print(f"File: {path}")
    print(f"Number of rows: {len(df)}")
    print(f"\n{'Column':<40} {'Dtype':<20} {'Shape'}")
    print("-" * 80)
    for col in df.columns:
        val = df[col].iloc[0]
        if hasattr(val, "shape"):
            shape = val.shape
        else:
            shape = "(scalar)"
        print(f"{col:<40} {str(df[col].dtype):<20} {shape}")

if __name__ == "__main__":
    main()
