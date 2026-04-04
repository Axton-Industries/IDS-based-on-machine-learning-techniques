import pandas as pd
import numpy as np


def get_non_downcastable_columns(df: pd.DataFrame):
    """
    Returns columns that cannot be safely downcasted.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    cannot_downcast_int = []
    cannot_downcast_float = []

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        if pd.api.types.is_integer_dtype(df[col]):
            fits_uint32 = min_val >= 0 and max_val <= np.iinfo(np.uint32).max
            fits_int32 = (
                min_val >= np.iinfo(np.int32).min
                and max_val <= np.iinfo(np.int32).max
            )

            if not (fits_uint32 or fits_int32):
                cannot_downcast_int.append(col)

        elif pd.api.types.is_float_dtype(series):
            # Range check
            in_range = (
                min_val >= np.finfo(np.float32).min
                and max_val <= np.finfo(np.float32).max
            )

            # Precision check (THIS is what you were missing)
            as_float32 = series.astype(np.float32)
            precision_ok = np.allclose(series, as_float32, rtol=1e-05, atol=1e-08)

            if not (in_range and precision_ok):
                cannot_downcast_float.append(col)

        print("Columnas enteras que NO se pueden downcast:")
        print(cannot_downcast_int)

        print("\nColumnas float que NO se pueden downcast:")
        print(cannot_downcast_float)

    return cannot_downcast_int, cannot_downcast_float


def downcast_numeric_columns(df: pd.DataFrame) -> None:
    """
    Downcasts numeric columns using the result from get_non_downcastable_columns.
    """
    cannot_int, cannot_float = get_non_downcastable_columns(df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col in cannot_int or col in cannot_float:
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= 0:
                df[col] = df[col].astype(np.uint32)
            else:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)

    print("\nDowncasting completado.")
