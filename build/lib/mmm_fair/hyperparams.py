from sklearn.tree import DecisionTreeClassifier
import numpy as np


def get_hparams(
    classifier: str = "MMM_Fair",
    dataset_name='Adult',
    constraint: str = "DP",
    alpha: float = 0.1,
    max_iter: int = 100,
    saIndex=None,
    saValue=None,
    random_state=None,
    # possibly other flags like 'pareto'
    **kwargs
):
    """
    Returns a dict of parameters to instantiate the desired classifier.
    :param classifier: "MMM_Fair" or "MMM_Fair_GBT"
    :param constraint: "DP", "EP", "EO", "TPR", or "FPR" 
    :param alpha: fairness weight
    :param max_iter: number of boosting iterations
    :param saIndex: sensitive array
    :param saValue: dictionary of protected groups
    :param random_state: for reproducibility
    :param kwargs: possibly more args (like store_iter_metrics, etc.)
    """
    if classifier.lower() in ["mmm_fair","mmm-fair","mmm"]:
        # Original AdaBoost-like version
        # We'll build a dictionary for MMM_Fair constructor
        
    
        dataset = dataset_name.lower()
        cstr = constraint.upper()
    
        # Set a fallback default
        mmm_params = {
            "estimator": DecisionTreeClassifier(max_depth=5, class_weight=None),
            "random_state": 0,
            "n_estimators": 500,
            "gamma": 0.25,
            "constraint": cstr,
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
            elif cstr == "EP" or cstr == "TPR":
                mmm_params.update({
                    "random_state": 0,
                    "n_estimators": 300,
                    "gamma": 0.25
                })
                pareto_bool = False
            elif cstr == "FPR":
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
            elif cstr == "EP" or cstr == "TPR":
                mmm_params.update({
                    "random_state": 0,
                    "n_estimators": 300,
                    "gamma": 0.25
                })
                pareto_bool = True
            elif cstr == "FPR":
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
    
    elif classifier.lower() in ["mmm_fair_gbt","mmm-fair-gbt","mmm_gbt", "mmm-gbt"]:
        # Our new gradient-based version
        mmm_params = {
            "constraint": constraint,
            "alpha": alpha,
            "saIndex": saIndex,
            "saValue": saValue,
            "max_iter": max_iter,
            "random_state": random_state,
            "store_iter_metrics": True,
            # etc. possibly early_stopping, validation_fraction, etc.
            "early_stopping":False, "validation_fraction": None,
        }
        pareto_bool = False
    
        return mmm_params, pareto_bool
        
    else:
        raise ValueError(f"Unknown classifier type '{classifier}'")