import pandas as pd
from ucimlrepo import fetch_ucirepo
from .mammoth_csv import CSV
from collections import OrderedDict


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

    if name == "adult":
        # Just an example if you want to transform e.g. "."
        label = label.str.replace(".", "", regex=False)


    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    csv_dataset.description = OrderedDict(
            [
                ("Summary", all_raw_data.metadata.additional_info.summary),
                ("Variables", all_raw_data.metadata.additional_info.variable_info),
            ]
        )
    return csv_dataset



def data_local(
    raw_data: pd.DataFrame,
    target: str = None
) -> CSV:
    
    numeric = [
        col
        for col in raw_data
        if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
        and len(set(raw_data[col])) > 10
    ]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least one categorical column is required.")
    if target is  None:
        if 'class' in categorical:
            target = 'class'
        elif 'Class' in categorical:
            taget = 'Class'
        elif 'Y' in categorical:
            target= 'Y'
        elif 'y' in categorical:
            target = 'y'
        else:
            target= categorical[-1]
    if target in categorical:
        categorical.remove(target)
    label = raw_data[target].copy()
    raw_data=raw_data.drop(columns=[target])
 

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    return csv_dataset

def data_raw(raw_data: pd.DataFrame = None,
    dataset_name: str = None,
    target=None,) -> CSV:

    try:
        if raw_data is None:
            if dataset_name is not None:
                if dataset_name.endswith(".csv"):
                    df = pd.read_csv(dataset_name)
                elif dataset_name.endswith(".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods"):
                    df = pd.read_excel(dataset_name)
                elif dataset_name.endswith(".json"):
                    df = pd.read_json(dataset_name)
                elif dataset_name.endswith(".html", ".htm"):
                    df = pd.read_html(dataset_name)
            else:
                raise ValueError("At least one of a dataset name or a raw dataframe must be provided.")
        else:
            df = raw_data

        csv_dataset = data_local(df, target)
        return csv_dataset
    except:
        raise ValueError("Could not read data. Unsupported or invalid format.")
