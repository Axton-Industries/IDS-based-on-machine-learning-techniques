import numpy as np
import pandas as pd


def shift_log_transform(df, feature_list, exclude_cols):
    """
    Apply shift + log1p transformation to selected features,
    excluding specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    feature_list : list
        List of candidate features to transform.
    exclude_cols : set or list
        Columns to exclude from transformation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with log-transformed features (prefixed with 'log_')
        and original transformed columns removed.
    """

    df_log = df.copy()

    cols_to_transform = [c for c in feature_list if c not in exclude_cols]

    for col in cols_to_transform:
        min_val = df_log[col].min()
        shift = 1 - min_val if min_val <= 0 else 0
        df_log[f"log_{col}"] = np.log1p(df_log[col] + shift)

    df_log = df_log.drop(columns=cols_to_transform)

    return df_log

