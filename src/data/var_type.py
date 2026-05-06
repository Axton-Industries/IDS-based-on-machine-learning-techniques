from typing import List, Tuple
import pandas as pd


def split_symbolic_continuous(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Splits DataFrame columns into symbolic and continuous features.

    Rules:
    - object/category dtype -> symbolic
    - binary features (nunique == 2) -> symbolic
    - special names: protocol, service, flag -> symbolic
    - everything else -> continuous

    Returns:
        symbolic (list of str), continuous (list of str)
    """

    symbolic: List[str] = []
    continuous: List[str] = []

    special_cols = {"protocol", "service", "flag"}

    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        col_lower = col.lower()

        # categorical types
        if df[col].dtype == "category" or df[col].dtype == "object":
            symbolic.append(col)

        # binary features
        elif unique_vals == 2:
            symbolic.append(col)

        # protocol-like features
        elif col_lower in special_cols:
            symbolic.append(col)

        else:
            continuous.append(col)

    return symbolic, continuous
