import pandas as pd
import numpy as np


INT32_INFO = np.iinfo(np.int32)
UINT32_INFO = np.iinfo(np.uint32)
FLOAT32_INFO = np.finfo(np.float32)


def fits_int32(min_val, max_val):
    return (
        min_val >= INT32_INFO.min
        and max_val <= INT32_INFO.max
    )


def fits_uint32(min_val, max_val):
    return (
        min_val >= 0
        and max_val <= UINT32_INFO.max
    )


def is_integer_like(series: pd.Series):
    s = series.dropna().to_numpy()

    return np.all(
        np.isclose(
            s,
            np.round(s),
            rtol=0,
            atol=1e-8
        )
    )


def get_non_downcastable_columns(df: pd.DataFrame):

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    cannot_downcast_int = []
    cannot_downcast_float = []

    for col in numeric_cols:

        series = df[col]

        if series.dropna().empty:
            continue

        s = series.dropna()

        min_val = s.min()
        max_val = s.max()

        # ==================================================
        # INTEGER COLUMNS
        # ==================================================

        if pd.api.types.is_integer_dtype(series):

            if not (
                fits_int32(min_val, max_val)
                or fits_uint32(min_val, max_val)
            ):
                cannot_downcast_int.append(col)

        # ==================================================
        # FLOAT COLUMNS
        # ==================================================

        elif pd.api.types.is_float_dtype(series):

            # -----------------------------
            # OPTION 1: float64 -> int32/uint32
            # -----------------------------

            integer_like = is_integer_like(series)

            fits_int = (
                fits_int32(min_val, max_val)
                or fits_uint32(min_val, max_val)
            )

            can_convert_to_int = integer_like and fits_int

            # -----------------------------
            # OPTION 2: float64 -> float32
            # -----------------------------

            in_float32_range = (
                min_val >= FLOAT32_INFO.min
                and max_val <= FLOAT32_INFO.max
            )

            precision_ok = np.allclose(
                s.to_numpy(),
                s.astype(np.float32).astype(np.float64),
                rtol=1e-5,
                atol=1e-8
            )

            can_convert_to_float32 = (
                in_float32_range
                and precision_ok
            )

            # -----------------------------
            # FINAL
            # -----------------------------

            if not (
                can_convert_to_int
                or can_convert_to_float32
            ):
                cannot_downcast_float.append(col)

    print("Columnas enteras que NO se pueden downcast:")
    print(cannot_downcast_int)

    print("\nColumnas float que NO se pueden downcast:")
    print(cannot_downcast_float)

    return cannot_downcast_int, cannot_downcast_float


def downcast_numeric_columns(df: pd.DataFrame) -> None:

    cannot_int, cannot_float = get_non_downcastable_columns(df)

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numeric_cols:

        if col in cannot_int or col in cannot_float:
            continue

        series = df[col]

        min_val = series.min()
        max_val = series.max()

        # ==================================================
        # INTEGER COLUMNS
        # ==================================================

        if pd.api.types.is_integer_dtype(series):

            if fits_uint32(min_val, max_val):
                df[col] = series.astype(np.uint32)
            else:
                df[col] = series.astype(np.int32)

        # ==================================================
        # FLOAT COLUMNS
        # ==================================================

        elif pd.api.types.is_float_dtype(series):

            # float -> int
            if is_integer_like(series):

                if fits_uint32(min_val, max_val):
                    df[col] = series.astype(np.uint32)

                elif fits_int32(min_val, max_val):
                    df[col] = series.astype(np.int32)

                else:
                    df[col] = series.astype(np.float32)

            # float -> float32
            else:
                df[col] = series.astype(np.float32)

    print("\nDowncasting completado.")
