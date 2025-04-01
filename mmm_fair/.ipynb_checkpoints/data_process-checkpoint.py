import pandas as pd
from ucimlrepo import fetch_ucirepo
from .mammoth_csv import CSV



def data_uci(
    dataset_name: str = None,
    d_id: int = None,
    target: str = None,
) -> CSV:
    name = dataset_name.lower()
    if name == "credit":
        d_id = 350
        if target is None:
            target = "Y"
    elif name == "bank":
        d_id = 222
        if target is None:
            target = "y"
    elif name == "adult":
        d_id = 2
        if target is None:
            target = "income"
    elif name == "kdd":
        d_id = 117
        if target is None:
            target = "income"
    else:
        # raise Exception("Unexpected dataset name: " + name)
        print("Unexpected dataset name: " + name)

    if d_id is None:
        all_raw_data = fetch_ucirepo(name)
    else:
        all_raw_data = fetch_ucirepo(id=d_id)
    raw_data = all_raw_data.data.features
    numeric = [
        col
        for col in raw_data
        if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
        and len(set(raw_data[col])) > 10
    ]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least two categorical columns are required.")
    label = all_raw_data.data.targets[target]

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    return csv_dataset
