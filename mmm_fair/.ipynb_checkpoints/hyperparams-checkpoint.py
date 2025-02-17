from sklearn.tree import DecisionTreeClassifier
import numpy as np

def get_hparams(dataset_name, constraint, data):
    """
    :param dataset_name: e.g. "adult", "bank", or something unknown.
    :param constraint: e.g. "DP", "EO", "EP".
    :param data: the CSV or DataFrame wrapper loaded from your data_uci or local file.

    :return: (mmm_params, pareto_bool)
    """
    dataset = dataset_name.lower()
    cstr = constraint.upper()

    # Set a fallback default
    mmm_params = {
        "estimator": DecisionTreeClassifier(max_depth=5, class_weight=None),
        "random_state": 0,
        "n_estimators": 500,
        "gamma": 0.25,
        "constraints": cstr,
        "saIndex": None,
        "saValue": None
    }
    pareto_bool = False

    # Known combos:
    if dataset == "adult":
        # (re)define any custom combos
        if cstr == "DP":
            mmm_params.update({
                "random_state": 42,
                "n_estimators": 250,
                "gamma": 0.5
            })
            pareto_bool = False
        elif cstr == "EO":
            mmm_params.update({
                "random_state": 0,
                "n_estimators": 1000,
            })
            pareto_bool = False
        elif cstr == "EP":
            mmm_params.update({
                "random_state": 0,
                "n_estimators": 300,
                "gamma": 0.25
            })
            pareto_bool = False
        else:
            # Unknown constraint on a known dataset, fallback to default
            pass



    elif dataset == "bank":
        if cstr == "EO":
            mmm_params.update({
                "random_state": 42,
                "n_estimators": 500,
            })
            pareto_bool = False
        elif cstr == "DP":
            mmm_params.update({
                "random_state": 42,
                "n_estimators": 400,
                "gamma": 0.5
            })
            pareto_bool = False
        elif cstr == "EP":
            mmm_params.update({
                "random_state": 0,
                "n_estimators": 300,
                "gamma": 0.25
            })
            pareto_bool = True
        else:
            pass

        # Setup protected attributes for BANK
        

    else:
        # Fallback for unknown dataset or local CSV
        print(f"Dataset '{dataset_name}' not recognized. Using default hyperparameters.")
        # If the user passes custom `prot_cols`, `nprotg_vals`, we can use them:
        

        # keep the default mmm_params set above
        pareto_bool = False

    
    return mmm_params, pareto_bool