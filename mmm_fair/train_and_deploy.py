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
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split



def build_sensitives(
    df: pd.DataFrame,
    protected_cols: list,
    non_protected_vals: list
):
    """
    Constructs saIndex and saValue for fairness analysis.

    :param df: A pandas DataFrame containing all relevant data.
    :param protected_cols: List of column names for protected attributes.
    :param non_protected_vals: A list (same length) of "non-protected" specs,
        which can be:
        - Single values (str, float, int) for categorical or exact numeric matching,
        - A (lower, upper) tuple for numeric range checks.
    :return: (saIndex, saValue)
        saIndex: a 2D NumPy array of shape (n_samples, len(protected_cols)).
                 Each column i is 1 if the row's value is "non-protected" for that attribute,
                 else 0 if "protected".
        saValue: a dictionary {protected_col: 0, ...} indicating the convention
                 that 0 is considered protected.
    """

    if len(protected_cols) != len(non_protected_vals):
        raise ValueError(
            f"Number of protected columns ({len(protected_cols)}) does not match "
            f"the number of non-protected values ({len(non_protected_vals)})."
        )

    # We'll store 1 if row is "non-protected" according to the definition, else 0
    saIndex = df[protected_cols].to_numpy(copy=True)  # shape (n_samples, #prot_cols)
    saValue = {col: 0 for col in protected_cols}      # By default, interpret 0 as "protected"

    # Iterate over each protected column with the corresponding "non-protected" definition
    for i, (col, nprot_val) in enumerate(zip(protected_cols, non_protected_vals)):

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            parsed_value = parse_numeric_input(nprot_val)
            # If numeric, see if nprot_val is a tuple => range, or single numeric => exact match
            if isinstance(parsed_value, tuple) and len(parsed_value) == 2:
                lower, upper = parsed_value
                # "Non-protected" if value is between lower and upper
                saIndex[:, i] = (
                    (saIndex[:, i].astype(float) > float(lower)) &
                    (saIndex[:, i].astype(float) < float(upper))
                ).astype(int)
            else:
                # Single numeric => direct match
                val_float = float(parsed_value)
                saIndex[:, i] = (saIndex[:, i].astype(float) == val_float).astype(int)

        else:
            # Categorical => simple string (or object) match
            saIndex[:, i] = (saIndex[:, i] == nprot_val).astype(int)

    return saIndex, saValue
    
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
    elif learner_str.lower() in ("xtree","extra", "extratree", "extra_tree"):
        return ExtraTreeClassifier( max_features = "sqrt")
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
    
    parser.add_argument("--deploy", type=str, default=None,
                        help="Deployment format: 'onnx' or 'pickle'.")
    parser.add_argument("--save_path", type=str, default="my_mmm_fair_model",
                        help="Path prefix for saved model(s).")
    parser.add_argument(
        "--base_learner", type=str, default='lr',
        help="Override the default estimator, e.g. 'tree', 'logistic', etc."
    )
    parser.add_argument(
        "--report_type", type=str, default='table',
        help="Override the default report output, e.g. 'table', 'console', 'html', etc."
    )
    parser.add_argument("--pareto", type=lambda x: x.lower() == 'true', default=False,
                    help="Set to True to select theta from ensembles with Pareto optimal solutions (default: False)")

    parser.add_argument("--test", type=str, default=None,
                        help="Either path to a test CSV or a float fraction (e.g. 0.3) for train/test split. If not provided, no separate testing is done.")
    args = parser.parse_args()
    dataset_name = None
    if args.dataset is not None:
        dataset_name = args.dataset.lower()

    # 1. Load data
    if dataset_name and dataset_name.endswith(".csv"):
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
    
    sensitives=[]
    for col, val in zip(args.prots, args.nprotgs):
        sensitives.append(col)
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
    # saIndex = data.data[args.prots].to_numpy()
    # saValue = {attr: 0 for attr in args.prots}
    # # For each column i, set 1 if it equals the non-protected value, else 0
    # for i, iprots in enumerate(zip(saValue,args.nprotgs)):
    #     if pd.api.types.is_numeric_dtype(data.data[col]):
    #         if isinstance(parsed_value, tuple):
    #             ((saIndex[:, i].astype(float) > parsed_value[0]) & (saIndex[:, i].astype(float) < parsed_value[1])).astype(int)
    #         else:
    #             saIndex[:, i] = (saIndex[:, i] == parsed_value).astype(int)
            
    #     else:
    #         saIndex[:, i] = (saIndex[:, i] == iprots[1]).astype(int)
    
    # By default, interpret 0 as 'protected'
    

    mmm_params, _ = get_hparams(
        dataset_name=args.dataset,
        constraint=args.constraint,
        data=data,
    )

    #mmm_params["saIndex"] = saIndex
    #mmm_params["saValue"] = saValue
    
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
    X = data.to_pred(sensitive=sensitives)  # or however you define

    
    #mmm_classifier = MMM_Fair(**mmm_params)

    # 5. Check if test data or split given
    if args.test is not None:
        # Could be a file or a float
        try:
            split_frac = float(args.test)
            if split_frac <= 0 or split_frac >= 1:
                raise ValueError("Train/Test split fraction must be between 0 and 1.")
            indices = np.arange(len(X))
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, indices, test_size=split_frac, random_state=42)    
            
            saIndex_train, saValue_train= build_sensitives(data.data.iloc[id_train], args.prots, args.nprotgs)
            saIndex_test, _= build_sensitives(data.data.iloc[id_test], args.prots, args.nprotgs)
            
            mmm_params["saIndex"] = saIndex_train
            mmm_params["saValue"] = saValue_train

            # 6. Construct MMM_Fair
            mmm_classifier = MMM_Fair(**mmm_params)
            mmm_classifier.fit(X_train, y_train)

        except ValueError:
            # If it's not a float, treat it as a CSV path
            test_df = pd.read_csv(args.test)
            # Minimal numeric/categorical detection, build CSV instance
            numeric_test = [col for col in test_df.columns if pd.api.types.is_numeric_dtype(test_df[col])]
            cat_test = [col for col in test_df.columns if col not in numeric_test and col != args.target]
            label_test = test_df[args.target]

            data_test = CSV(test_df, numeric=numeric_test, categorical=cat_test, labels=label_test)            
            
            if not list(data.data.columns) == list(data_test.data.columns):
                raise ValueError(
                    "Mismatch between train and test columns!\n"
                    f"Train columns: {list(df.columns)}\n"
                    f"Test columns: {list(df_test.columns)}"
                )
            X_test = data_test.to_pred(sensitive=sensitives)
            y_test = data_test.labels["label"].to_numpy()
            #y_pred = mmm_classifier.predict(X_test)
            #y_true=y_test
            # Train on entire main data
            saIndex, saValue= build_sensitives(data.data, args.prots, args.nprotgs)
            saIndex_test, _= build_sensitives(data_test.data, args.prots, args.nprotgs)
            
            mmm_params["saIndex"] = saIndex
            mmm_params["saValue"] = saValue
            
            # 6. Construct MMM_Fair
            mmm_classifier = MMM_Fair(**mmm_params)
            
            mmm_classifier.fit(X, y)
    else:
        saIndex, saValue= build_sensitives(data.data, args.prots, args.nprotgs)
        saIndex_test=saIndex
        mmm_params["saIndex"] = saIndex
        mmm_params["saValue"] = saValue
        # 6. Construct MMM_Fair
        mmm_classifier = MMM_Fair(**mmm_params)
        mmm_classifier.fit(X, y)
        #test on the training data
        X_test, y_test= X, y


    # 6. Pareto setting
    mmm_classifier.pareto = args.pareto 
    mmm_classifier.update_theta(criteria="all")

    # 7. (Optional) FairBench reporting
    y_pred = mmm_classifier.predict(X_test)
    # If you only want the first protected col, do e.g. saIndex[:,0]
    # or otherwise combine them
    if args.report_type.lower()=="table":
        rt=fb.export.ConsoleTable
    elif args.report_type.lower()=="console":
        rt=fb.export.Console
    elif args.report_type.lower()=="html":
        rt=fb.export.Html(horizontal=False, view=False)
    else:
        print(f"Report type {args.report_type} not supported in this version. Switching to table type reporting.")
        rt=fb.export.ConsoleTable
    for i in range(len(sensitives)):
        print("Reports generated for protected attribute ", mmm_classifier.sensitives[i])
        sens = fb.categories(saIndex_test[:, i])
        report = fb.reports.pairwise(
            predictions=y_pred,
            labels=y_test,
            sensitive=sens
        )
        
        out=report.show(env=rt)
        

    # 8. Deployment
    if args.deploy not in ("onnx", "pickle"):
        # If user didn't provide or gave something unrecognized => prompt
        while True:
            user_choice = input("\nNo valid deployment option provided.\n"
                                "Enter '1' for ONNX, '2' for pickle, '3' to exit: ").strip()
            if user_choice == '1':
                args.deploy = "onnx"
                break
            elif user_choice == '2':
                args.deploy = "pickle"
                break
            elif user_choice == '3':
                print("Exiting without saving model.")
                return
            else:
                print("Invalid input. Please try again (1, 2, or 3).")

    # Now we have a recognized deploy format or user has chosen
    if args.deploy == "onnx":
        convert_to_onnx(mmm_classifier, args.save_path, X_full)
        print(f"Model saved in ONNX format with prefix '{args.save_path}'")
    elif args.deploy == "pickle":
        convert_to_pickle(mmm_classifier, args.save_path)
        print(f"Model saved in pickle format as '{args.save_path}.pkl'")

if __name__ == "__main__":
    main()