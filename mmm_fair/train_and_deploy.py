import argparse
import numpy as np
import pandas as pd
import fairbench as fb

from sklearn.tree import DecisionTreeClassifier

# local imports
from .data_process import data_uci
from .mammoth_csv import CSV
from .mmm_fair import MMM_Fair
from .deploy_utils import convert_to_onnx, convert_to_pickle
from .hyperparams import get_hparams  # The function that sets hyperparams or fallback
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def parse_numeric_input(value_str):
    """
    Parses a numeric input which can be:
    - A single numeric value: "30" -> 30.0
    - A range of two values: "30_60" -> (30.0, 60.0)

    Returns either:
    - A tuple (lower, upper) if it's a range.
    - A single float/int if it's just one number.
    """
    try:
        if "_" in value_str:
            lower, upper = map(float, value_str.split("_"))
            return lower, upper  # Return as a tuple for range
        else:
            return float(value_str)  # Return single numeric value
    except ValueError:
        raise ValueError(f"Invalid numeric format '{value_str}'. Expected a single number '30' or a range '30_60'.")

# --- Processing logic ---
        
def parse_base_learner(learner_str):
    if learner_str.lower() in ("tree","dt", "decisiontree", "decision_tree"):
        return DecisionTreeClassifier(max_depth=5, class_weight=None)
    elif learner_str.lower() in ("logistic", "logreg","lr"):
        return LogisticRegression(max_iter=1000)
    # elif learner_str.lower() in ("mlp","nn"):
    #     return  MLPClassifier()
    else:
        raise ValueError(f"Unrecognized base_learner: {learner_str}")
        
def main():
    parser = argparse.ArgumentParser(description="Train and Deploy MMM_Fair model")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of dataset or path to a local CSV file.")
    parser.add_argument("--target", type=str, default=None,
                        help="Label column if using known dataset or CSV.")
    parser.add_argument("--pos_Class", type=str, default=None,
                        help="Positive class Label if using known dataset or CSV.")
    parser.add_argument(
        "--prots", 
        nargs="+", 
        default=[], 
        help="List of protected attribute names (e.g. --prots race sex age)."
    )
    # Similarly, a list of non-protected values for each attribute:
    parser.add_argument(
        "--nprotgs", 
        nargs="+", 
        default=[], 
        help="List of non-protected attribute values, matching order of --prots."
    )

    parser.add_argument("--constraint", type=str, default="EO",
                        help="Fairness constraint: DP, EO, or EP.")
    
    parser.add_argument("--deploy", type=str, default="onnx",
                        help="Deployment format: 'onnx' or 'pickle'.")
    parser.add_argument("--save_path", type=str, default="my_mmm_fair_model",
                        help="Path prefix for saved model(s).")
    parser.add_argument(
        "--base_learner", type=str, default='lr',
        help="Override the default estimator, e.g. 'tree', 'logistic', etc."
    )

    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    # 1. Load data
    if dataset_name.endswith(".csv"):
        # -------------------------
        # Local CSV file fallback
        # -------------------------
        raw_df = pd.read_csv(args.dataset)
        
        # Minimal auto-detection of numeric vs. categorical
        numeric = [col for col in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[col])]
        categorical = [col for col in raw_df.columns if col not in numeric and col != args.target]
        
        # Create the CSV object from mammoth_csv.py
        # We assume user-supplied --target is in raw_df
        label = raw_df[args.target]
        
        data = CSV(
            raw_df,
            numeric=numeric,
            categorical=categorical,
            labels=label,
        )
    else:
        # -------------------------
        # Known dataset (Adult, Bank, etc.)
        # via data_uci function
        # -------------------------
        data = data_uci(dataset_name=args.dataset, target=args.target)

    # 2. Retrieve hyperparameters & fallback for unknown data
   
    # -- Validate the prot_cols and nprotg_vals lengths match
    if len(args.prots) != len(args.nprotgs):
        raise ValueError(
            f"Number of protected attributes ({len(args.prots)}) "
            f"doesn't match number of non-protected values ({len(args.nprotgs)}). "
            f"Please provide them in pairs."
        )
    
    for col, val in zip(args.prots, args.nprotgs):
        if col not in data.data.columns:
            raise ValueError(
                f"Protected attribute '{col}' is not a valid column. "
                f"Available columns: {list(data.data.columns)}"
            )
        else:
            if pd.api.types.is_numeric_dtype(data.data[col]):
                parsed_value = parse_numeric_input(val)
                if isinstance(parsed_value, tuple): 
                    if parsed_value[0] < data.data[col].min() or parsed_value[1] > data.data[col].max():
                        raise ValueError(
                            f"{col} range '{val}' is outside dataset range [{data.data[col].min()}, {data.data[col].max()}]."
                        )
                else:  # If it's a single numeric value
                    if parsed_value < data.data[col].min() or parsed_value > data.data[col].max():
                        raise ValueError(
                            f"Numeric value '{val}' is outside dataset range [{data.data[col].min()}, {data.data[col].max()}]."
                        )
                    
            else:
                unique_vals = data.data[col].unique()
                if val not in unique_vals:
                    raise ValueError(
                        f"Value '{val}' not found in column '{col}'. "
                        f"Unique values are: {unique_vals}"
                    )
    saIndex = data.data[args.prots].to_numpy()
    saValue = {attr: 0 for attr in args.prots}
    # For each column i, set 1 if it equals the non-protected value, else 0
    for i, iprots in enumerate(zip(saValue,args.nprotgs)):
        if pd.api.types.is_numeric_dtype(data.data[col]):
            if isinstance(parsed_value, tuple):
                ((saIndex[:, i].astype(float) > parsed_value[0]) & (saIndex[:, i].astype(float) < parsed_value[1])).astype(int)
            else:
                saIndex[:, i] = (saIndex[:, i] == parsed_value).astype(int)
            
        else:
            saIndex[:, i] = (saIndex[:, i] == iprots[1]).astype(int)
    
    # By default, interpret 0 as 'protected'
    

    mmm_params, pareto_bool = get_hparams(
        dataset_name=args.dataset,
        constraint=args.constraint,
        data=data,
    )

    mmm_params["saIndex"] = saIndex
    mmm_params["saValue"] = saValue
    
    if args.base_learner is not None:
        print(f"Loading MMM-Fair with base learner: {args.base_learner}")
        mmm_params["estimator"] = parse_base_learner(args.base_learner)
    # 3. Convert label array if needed
    y = data.labels["label"].to_numpy()
    if args.dataset.lower() == "adult":
        # Just an example if you want to transform e.g. "."
        y = np.array([s.replace(".", "") for s in y])
        # Possibly recast to 0/1 if you want:
    pos_class=args.pos_Class  
    if pos_class not in list(set(y)):
        pos_class=list(set(y))[0]
    y = (y == pos_class).astype(int)

    # 4. Get feature matrix (some users do data.to_pred([...]) or data.to_features([...]))
    X = data.to_pred(sensitive=saValue.keys())  # or however you define

    # 5. Construct MMM_Fair
    mmm_classifier = MMM_Fair(**mmm_params)
    mmm_classifier.fit(X, y)

    # 6. Pareto setting
    mmm_classifier.pareto = pareto_bool
    mmm_classifier.update_theta(criteria="all")

    # 7. (Optional) FairBench reporting
    y_pred = mmm_classifier.predict(X)
    # If you only want the first protected col, do e.g. saIndex[:,0]
    # or otherwise combine them
    for i in range(len(saValue)):
        print("Reports generated for protected attribute ", mmm_classifier.sensitives[i])
        sens = fb.categories(saIndex[:, i])
        report = fb.reports.pairwise(
            predictions=y_pred,
            labels=y,
            sensitive=sens
        )
        #report.show(env=fb.export.ConsoleTable)
        report.show(env=fb.export.Console)
        html_text = report.show(fb.export.Html(horizontal=False, view=False)) 

    # 8. Deployment
    if args.deploy == "onnx":
        convert_to_onnx(mmm_classifier, args.save_path, X)
    else:
        convert_to_pickle(mmm_classifier, args.save_path)

if __name__ == "__main__":
    main()