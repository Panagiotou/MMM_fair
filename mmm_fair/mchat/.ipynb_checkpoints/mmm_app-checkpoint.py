import os
from flask import Flask, render_template, request, session, jsonify
from flask_session import Session
import argparse
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from flask import send_from_directory
# Import required modules
from mmm_fair.train_and_deploy import (
    generate_reports, 
    parse_base_learner,
    parse_numeric_input,
    build_sensitives,
    train
)
from mmm_fair.deploy_utils import convert_to_onnx
from mmm_fair.viz_trade_offs import plot2d, plot3d
from mmm_fair.mmm_fair import MMM_Fair
from mmm_fair.mmm_fair_gb import MMM_Fair_GradientBoostedClassifier
from mmm_fair.data_process import data_uci  # Importing data processing module
import os
import uuid
import plotly.io as pio

# Ensure 'static/plots' directory exists
PLOT_DIR = "mmm_fair/mchat/static/"
os.makedirs(PLOT_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
app.secret_key = "SOME_SECRET"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Define chat arguments
CHAT_ARGS = [
    "classifier",
    "dataset",
    "prots",
    "nprotgs",
    "target",
    "pos_Class",
    "n_learners",
    "constraint",
    "save_as",
    "moo_vis"
]

default_args = {
    "classifier": "MMM_Fair_GBT",
    "dataset": 'Adult',
    "target": 'income',
    "pos_Class": '>50K',
    "n_learners": 100,
    "prots": ['race', 'sex'],
    "nprotgs": ['White', 'Male'],
    "constraint": "EO",
    "save_as": 'Onnx',
    "save_path": "my_mmm_fair_model",
    "base_learner": None,
    "report_type": "table",
    "pareto": False,
    "test": 0.3,
    "early_stop": False,
    "moo_vis": True  
}

DEFAULT_BOT_MESSAGES = [
    {"sender": "bot", "text": "Hello! Welcome to MMM-Fair Chat.\n\n"
                              "I can help you train either:\n"
                              " - 'MMM_Fair' (the AdaBoost style), or\n"
                              " - 'MMM_Fair_GBT' (the Gradient Boosting style).\n\n"
                              "Please type 'MMM_Fair' or 'MMM_Fair_GBT' to begin. \n"
                              "Or type 'default' to run with default parameters."}
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session.clear()
    session["chat_history"] = list(DEFAULT_BOT_MESSAGES)
    session["user_args"] = {}
    session["plot_all_url"]=''
    return jsonify({"ok": True})

@app.route("/ask_chat", methods=["POST"])
def ask_chat():
    chat_history = session.get("chat_history", [])
    user_args = session.get("user_args", {})
    user_msg = request.json.get("message", "").strip()

    # **NEW FEATURE**: If user enters "default", skip everything and run with default_args
    if not user_args and user_msg.lower() == "default":
        for arg in default_args:
            if arg not in user_args:
                user_args[arg]=default_args[arg]
        session["classifier"]=user_args["classifier"]
        chat_history.append({"sender": "bot", "text": "Running MMM_Fair with default parameters..."})
        vis_all, vis_fair = run_mmm_fair_app(default_args)
        #chat_history.append({"sender": "bot", "text": "Training complete with default parameters!"})
        #session["chat_history"] = chat_history
        #return jsonify({"chat_history": chat_history[-1:]})
        
        return jsonify({
        "chat_history": [{"sender": "bot", "text": "Here is the Pareto front. Select a Theta index."}],
        "plot_all_url": vis_all,
        "plot_fair_url": vis_fair
            })

    # Add user message
    if user_msg:
        chat_history.append({"sender": "user", "text": user_msg})

    missing_args = get_missing_args(user_args)

    if not missing_args:
        if user_msg.lower() == "run":
            for arg in default_args:
                if arg not in user_args:
                    user_args[arg]=default_args[arg]
            session["classifier"]=user_args["classifier"]
            vis_all, vis_fair = run_mmm_fair_app(user_args)
            #chat_history.append({"sender": "bot", "text": result_msg})
            return jsonify({
                    "chat_history": [{"sender": "bot", "text": "Here is the Pareto front. Select a Theta index."}],
                    "plot_all_url": vis_all,
                    "plot_fair_url": vis_fair
                })
        else:
            chat_history.append({
                "sender": "bot",
                "text": "We have all arguments. Type 'run' to start training or 'reset' to start over."
            })
    else:
        current_arg = missing_args[0]
        valid, clean_val, err_msg = validate_arg(current_arg, user_msg, user_args)

        # **NEW FEATURE**: Load dataset after fetching "dataset" argument
        if current_arg == "dataset":
            
            if valid:
                # Load dataset and store for later use
                try:
                    session["data"] = data_uci(clean_val)  # Load dataset
                    user_args[current_arg] = clean_val
                    chat_history.append({"sender": "bot", "text": f"Set {current_arg} = {clean_val}."})
                    chat_history.append({"sender": "bot", "text": "Dataset loaded successfully."})
                    
                except Exception as e:
                    chat_history.append({"sender": "bot", "text": f"Error loading dataset: {str(e)}\nPlease enter a valid dataset name."})
                    session["chat_history"] = chat_history
                #return jsonify({"chat_history": chat_history[-1:]})

            else:
                chat_history.append({"sender": "bot", "text": f"Error: {err_msg}\nPlease re-enter {current_arg}."})
                session["chat_history"] = chat_history
                #return jsonify({"chat_history": chat_history[-1:]})

        else:
            #valid, clean_val, err_msg = validate_arg(current_arg, user_msg, user_args)
            if not valid:
                chat_history.append({"sender": "bot", "text": f"Error: {err_msg}\nPlease re-enter {current_arg}."})
            else:
                user_args[current_arg] = clean_val
                chat_history.append({"sender": "bot", "text": f"Set {current_arg} = {clean_val}."})
        new_missing = get_missing_args(user_args)
        if new_missing:
            next_arg = new_missing[0]
            prompt = get_prompt_for_arg(next_arg, user_args)
            chat_history.append({"sender": "bot", "text": prompt})
        else:
            chat_history.append({"sender": "bot",
                "text": "All arguments captured. Type 'run' to train, or 'reset' to start over."})
        

    session["chat_history"] = chat_history
    session["user_args"] = user_args
    return jsonify({"chat_history": chat_history[-1:]})


def get_missing_args(user_args):
    """
    Return the list of arguments we still need, considering 
    if user chose MMM_Fair_GBT => skip base_learner
    """
    chosen_classifier = user_args.get("classifier", "").lower()

    if not chosen_classifier:
        return ["classifier"]  # No classifier => we can't skip anything yet

    needed_args = []
    for arg in CHAT_ARGS:
        if arg == "base_learner" and chosen_classifier in ["mmm_fair_gbt", "mmm-fair-gbt"]:
            continue  # Skip base_learner for GBT models
        if arg not in user_args:
            needed_args.append(arg)
    
    return needed_args

def validate_arg(arg_name, user_input, user_args):
    """
    Simple validator that can parse or store defaults.
    Returns (valid, clean_value, error_message).
    """
    if arg_name == "classifier":
        val = user_input.lower()
        if val not in ["mmm_fair", "mmm_fair_gbt", "mmm-fair", "mmm-fair-gbt"]:
            return False, None, "Classifier must be 'MMM_Fair' or 'MMM_Fair_GBT'."
        return True, "MMM_Fair_GBT" if "gbt" in val else "MMM_Fair", ""

    elif arg_name == "n_learners":
        if not user_input.isdigit():
            return False, None, "n_learners must be an integer e.g. 100"
        return True, int(user_input), ""

    elif arg_name == "constraint":
        c = user_input.upper()
        if c not in ["DP", "EP", "EO"]:
            return False, None, "Constraint must be DP, EP, or EO."
        return True, c, ""

    elif arg_name == "dataset":
        if user_input.lower().endswith(".csv"):
            return True, user_input, ""
        elif user_input.lower() in ["adult", "bank", "kdd", "credit"]:
            return True, user_input.lower(), "Loading Data...please wait!!!"
        else:
            return False, user_input, ""  # Fallback case

    elif arg_name == "target":
        # Extract the name of the target column
        available_target_name = session["data"].labels['label'].name  # Get the column name
    
        #print(f"DEBUG: Available target name: {available_target_name}")  # Debugging output
    
        return True, user_input, ""

    elif arg_name == "pos_Class":
        if not user_input:
            return True, None, ""  # Default None
        return True, user_input, ""

    elif arg_name == "prots":
        available_columns = session["data"].data.columns.tolist()
        available_columns=[v.lower() for v in available_columns]
        if not user_input:
            return False, [], f"At least one protected attribute expected"
        else:
            for prot in user_input.split(" "):
                if prot.lower() not in available_columns:
                    return False, None, f"Invalid protected. Available list of attributes in the data: {available_target_name}."
                
            return True, user_input.split(), ""

    elif arg_name == "nprotgs":
        existing_prots = user_args.get("prots", [])
        splitted = user_input.split()
        if len(splitted) != len(existing_prots):
            return False, None, f"Got {len(splitted)} non-protected vals for {len(existing_prots)} protected columns. Please match count."
        else:
            for i in range(len(splitted)):
                column=session["data"].data[existing_prots[i]]
                if pd.api.types.is_numeric_dtype(column):
                    parsed_value = parse_numeric_input(splitted[i])
                    if isinstance(parsed_value, tuple): 
                        if parsed_value[0] < column.min() or parsed_value[1] > column.max():
                            return False, None,f"Numeric non-protected input is outside dataset range [{column.min()}, {column.max()}]."
                    else:  # If it's a single numeric value
                        if parsed_value < column.min() or parsed_value > column.max():
                            return False, None,f"Numeric non-protected input is outside dataset range [{column.min()}, {column.max()}]."

                else:
                    unique_vals = column.unique()
                    unique_vals=[v.lower() for v in unique_vals]
                    if splitted[i].lower() not in unique_vals:
                        #raise ValueError(f"{splitted[i]}We are here{unique_vals}")
                        return False, None, f"Non-protected category must be in {unique_vals}"
            return True, user_input.split(), ""

    elif arg_name == "deploy":
        val = user_input.lower()
        if val not in ["onnx", "pickle"]:
            return False, None, "Deployment must be 'onnx' or 'pickle'."
        return True, val, ""

    elif arg_name == "moo_vis":
        if user_input.lower() in ["true", "yes", "y"]:
            return True, True, ""
        else:
            return True, False, ""

    else:
        return True, user_input, ""  # Default case
        
def get_prompt_for_arg(arg_name, user_args):
    """
    Return a question/prompt for the user 
    based on arg_name
    """
    prompts = {
        "dataset": "Enter dataset name from uci library ('adult','bank') or CSV path (e.g. data.csv):",
        "target": "Enter the label (target) column in your dataset (e.g. 'income'):",
        "pos_Class": "Enter the positive class label if known (else press Enter to skip):",
        "n_learners": "How many learners / iterations? e.g. '100'",
        "constraint": "Which fairness constraint? (DP, EP, or EO)?",
        "prots": "Enter space-separated protected attributes (e.g. 'race sex age'), or press Enter to skip if none:",
        "nprotgs": "Enter corresponding non-protected spec, e.g. 'White Male 30_60' matching the above columns:",
        "deploy": "Would you like to deploy as 'onnx' or 'pickle'?",
        "moo_vis": "Do you want to enable multi-objective visualization? 'True' or 'False'?",
    }

    if arg_name=="classifier":
        return ("Please pick 'MMM_Fair' (AdaBoost) or 'MMM_Fair_GBT' (gradient boosting):")
    if arg_name=="base_learner":
        return ("Enter base learner (tree, lr, logistic, extratree, etc.):")

    return prompts.get(arg_name, f"Enter {arg_name}:")

@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    session.pop("user_args", None)
    session.pop("data", None)
    return "Chat reset done."

def run_mmm_fair_app(user_args):
    args = argparse.Namespace(**user_args)
    mmm_classifier, X_test, y_test, saIndex_test, sensitives=train(args)
    session["mmm_classifier"] = mmm_classifier
    session["xtest"]=X_test
    session["ytest"]=y_test
    PF=np.array([mmm_classifier.ob[i] for i in range(len(mmm_classifier.ob))])
    thetas=np.arange(len(mmm_classifier.ob))
    title=f"3D Scatter Plot. Showing various trade-off points between Accuracy, Balanced Accuracy, and Maximum violation of {mmm_classifier.constraint} fairness among protected attributes."
    vis_all=plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria="Multi",
           axis_names=['Acc.','Balanc. Acc', 'MMM-fair'],title=title, html=True)
    PF=np.array([mmm_classifier.fairobs[i] for i in range(len(mmm_classifier.fairobs))])
    title=f"3D Scatter Plot. Showing various trade-off points between maximum violation of Demopgraphic Parity, Equal Opportunity, and Equalized odds fairness for the given set of protected attributes."
    vis_fair=plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria= "Multi-definitions",
           axis_names=['DP','EqOpp', 'EqOdd'],title=title,html=True)

    plot_all = f"all_.html"
    plot_fair = f"fair_.html"

    plot_all_path = os.path.join(PLOT_DIR, plot_all)
    plot_fair_path = os.path.join(PLOT_DIR, plot_fair)

    # Save the Plotly-generated HTML directly to files
    with open(plot_all_path, "w") as f:
        f.write(vis_all)
    with open(plot_fair_path, "w") as f:
        f.write(vis_fair)

    # Return URLs to the generated HTML plots
    return f"/static/{plot_all}", f"/static/{plot_fair}"

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory("static", filename)

@app.route('/static/<path:filename>')
def serve_plot(filename):
    plot_dir = os.path.abspath("static/plots")  # Ensure absolute path
    return send_from_directory(plot_dir, filename)   


@app.route("/update_model", methods=["POST"])
def update_model():
    # Get Theta value from the request
    data = request.json
    theta_index = int(data.get("theta", -1))

    # Retrieve trained classifier
    mmm_classifier = session.get("mmm_classifier")
    if not mmm_classifier:
        return jsonify({"success": False, "error": "No trained model found! Run training first."})

    # Validate Theta index
    if theta_index < 0 or theta_index >= len(mmm_classifier.ob):
        return jsonify({"success": False, "error": f"Invalid Theta index. Please select between 0 and {len(mmm_classifier.ob) - 1}."})

    # Update the model with selected Theta
    mmm_classifier.update_theta(theta=theta_index)
    session["mmm_classifier"] = mmm_classifier  # Store updated model

    return jsonify({"success": True, "message": f"Model updated with Theta index {theta_index}."})

@app.route("/save_model", methods=["POST"])
def save_model():
    data = request.json
    save_path = data.get("save_path", "").strip()

    # Retrieve trained classifier
    mmm_classifier = session.get("mmm_classifier")
    clf= session.get("classifier")
    xdata=session.get("xtest")
    user_args = session.get("user_args", {})

    if not mmm_classifier:
        return jsonify({"success": False, "error": "No trained model found! Run training first."})

    # Validate save path
    if not save_path or not os.path.isdir(save_path):
        return jsonify({"success": False, "error": "Invalid directory. Please select a valid folder."})

    # Call deploy() to save the model in the user-specified directory
    try:
        user_args["save_path"] = save_path  # Update args with the user-selected path
        convert_to_onnx( mmm_classifier, save_path, xdata, clf)
        return jsonify({"success": True, "message": f"Model saved in {save_path}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()

