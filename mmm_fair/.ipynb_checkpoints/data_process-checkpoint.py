import pandas as pd
from ucimlrepo import fetch_ucirepo
from .mammoth_csv import CSV

from sklearn import preprocessing
import numpy as np
'''
def preprocess(df, y, categorical, numerical, sensitive,  sex='', race='', age='', marg=''):
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])
    
    """ Feature normalization and one hot encoding """
    # convert class label 0 to -1
    #y = df[classf].copy(deep=True)
    y =  np.array([int(k) for k in y])
    y[y==0] = -1
    X=df.copy(deep=True)#.drop(columns=['ID', classf])
    if 'ID' in df.columns:
        X.drop(columns=['ID'], inplace=True)
    p_group=[0,0,0]
    #X= np.array([]).reshape(len(y), 0)
    for attr in X.columns: 
        vals = X[attr].copy()
        if attr in categorical:
            X[attr] = X[attr].fillna('missing')
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
        elif attr in numerical:
            vals = X[attr]
            if attr==age:
                va=[v for v in vals] 
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            
            if attr==age:
                v=[vals[i][0] for i in range(len(vals)) if va[i]==25 or va[i]==60]
                v=list(set(v))
                #print(v)
                p_group[sensitive.index(age)]=[min(v),max(v)]         
        X[attr]=vals
    return X, y, p_group
'''

def data_uci(
    dataset_name: str = None,d_id: int =None,
    target: str=None,
) -> CSV:

    name = dataset_name.lower()
    if name == "credit":
        d_id = 350
        if target==None:
            target='Y'
    elif name == "bank":
        d_id = 222
        if target==None:
            target='y'
    elif name=="adult":
        d_id = 2
        if target==None:
            target='income'
    elif name=="kdd":
        d_id=117
        if target==None:
            target='income'
    else:
        #raise Exception("Unexpected dataset name: " + name)
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