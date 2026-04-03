import pandas as pd
from pathlib import Path

def load_and_merge_csvs(dir: str | Path) -> pd.DataFrame:
    """
    Load all CSV files from a directory and merge them into a single DataFrame.

    Args:
        dir (str | Path): Path to the folder containing raw CSV files.

    Returns:
        pd.DataFrame: Merged DataFrame containing all rows from all CSVs.
    """
    dir = Path(dir)
    if not dir.exists() or not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir}")

    csv_files = list(dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dir}")

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = file.stem
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df
