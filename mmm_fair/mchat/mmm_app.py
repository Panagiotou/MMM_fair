import os
import time
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
from mmm_fair.mchat.langchain_utils import (get_langchain_agent, 
                                    summarize_html_report, 
                                    get_available_llm_providers)
from mmm_fair.mchat.openai_utils import get_openAI_llm
from mmm_fair.mchat.groq_utils import get_groq_llm

import uuid
import plotly.io as pio
llm=None

BASE_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(PLOT_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=PLOT_DIR)
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
    "constraint",
    
    "n_learners",
    #"save_as",
    #"moo_vis"
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
    session["llm_enabled"] = False
    session["api_enabled"] = False
    return render_template("index.html")
    
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    full_path = os.path.join(PLOT_DIR, filename)
    print(f"Serving static file: {full_path}")
    return send_from_directory(PLOT_DIR, filename)

@app.route("/provide_api_key", methods=["POST"])
def provide_api_key():
    try:
        env_key_map = {
                "openai": "OPENAI_API_KEY",
                "chatgpt": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "groqai": "GROQ_API_KEY",
                "together": "TOGETHER_API_KEY"
            }
        data = request.json
        api_key = data.get("api_key", "").strip()
        model = data.get("model", "").strip()
        if not api_key:
            return jsonify({"success": False, "error": "No API key provided."})

        if not model:
            return jsonify({"success": False, "error": "No LLM provider specified."})
        
        env_var = env_key_map.get(model.lower())
        if not env_var:
            return jsonify({"success": False, "error": f"Unknown provider '{model}'."})
            
        os.environ[env_var] = api_key
        session["api_enabled"] = True

        return jsonify({"success": True, "message": "✅ API key accepted. LLM features are now enabled."})

    except ImportError:
        return jsonify({
            "success": False,
            "error": "LangChain/OpenAI module not installed. Install with: pip install mmm-fair[llm-gpt]"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error occurred: {str(e)}"
        })


@app.route("/start_chat", methods=["POST"])
def start_chat():
    session.clear()
    #session["chat_history"] = list(DEFAULT_BOT_MESSAGES)
    session["user_args"] = {}
    session["plot_all_url"]=''
    if session.get("llm_enabled"):      
        template=[("system","You are an assistant inside the MMM-Fair Chat."),
                  ("User", "Greet the user politely and explain that this tool can train fairness-aware models.Use the following context: {context}")]
        llm = get_openAI_llm()
        llm_agent=get_langchain_agent(llm,template)
        message = langchain_agent.invoke({"context": DEFAULT_BOT_MESSAGES[0]["text"]})
        session["chat_history"] = [{"sender": "bot", "text": message}]
    else:
        session["chat_history"] = list(DEFAULT_BOT_MESSAGES)
    return jsonify({"ok": True})

def prompt_for_llm_provider(chat_history):
    available = get_available_llm_providers()
    session["valid_providers"] = available
    if not available:
        chat_history.append({
            "sender": "bot",
            "text": "⚠️ No LLM providers are available. Please install with:\n"
                    "`pip install mmm-fair[llm-gpt]`, `[llm-groq]`, or `[llm-together]`"
        })
    else:
        session["llm_enabled"]=True
        
        chat_history.append({
            "sender": "bot",
            "text": f"🧠 Please type the name of one of the available providers for explanation: {', '.join(list(available.keys())[::2])}"
        })
    return chat_history


def handle_llm_provider_selection(user_msg, chat_history):
    user_selection = user_msg.lower()
    valid_providers = session.get("valid_providers")
    if user_selection in valid_providers:
        session["llm_provider"] = user_selection
        chat_history.append({
            "sender": "bot",
            "text": f"✅ Provider '{user_selection}' selected. Please type 'yes' again to continue."
        })
    else:
        chat_history.append({
            "sender": "bot",
            "text": f"⚠️ Unknown provider '{user_selection}'. Please choose one of: {', '.join(list(valid_providers.keys()))}"
        })
    return chat_history
#"openai", "chatgpt", "groqai", "groq", "togetherai", "together"
@app.route("/ask_chat", methods=["POST"])
def ask_chat():
    chat_history = session.get("chat_history", [])
    user_args = session.get("user_args", {})
    user_msg = request.json.get("message", "").strip()

    if session.get("last_action") == "plotted" and user_msg.lower() in ["yes", "explain", "please explain", "what do these mean?", "openai", "chatgpt", "groqai", "groq", "togetherai", "together"]:
        if not session.get("llm_enabled"):
            session["llm_provider"] = "in-progress"
            chat_history = prompt_for_llm_provider(chat_history)
            session["chat_history"] = chat_history
            
            return jsonify({"chat_history": chat_history[-1:]})
        elif session.get("llm_provider") and len(session.get("valid_providers"))>0 : 
            if session.get("llm_provider")== "in-progress":
                chat_history = handle_llm_provider_selection(user_msg, chat_history)
                return jsonify({"chat_history": chat_history[-1:]})
            else:
                if not session.get("api_enabled"):
                    return jsonify({"require_api_key": True, "provider": session.get("llm_provider")})
                try:
                    vis_all=session.get("plot_all_url")
                    vis_fair=session.get("plot_fair_url")
                    vis_table=session.get("plot_table_path")
                    providers= session.get("valid_providers")
                    selected_model = session.get("llm_provider")
                    llm = providers[selected_model]()
                    template=[("system", "You are a fairness analysis expert, specially skilled to point out trade-offs between different performance metrics and fairness metrics, and gives suggestive guidance to user where fairness needs to be improved."),
                    ("user", "Please explain in brief summary the following given report  focusing mainly onthe most important numbers about fairness and predictive performance across protected attributes:\n{context}\n\n"
                     )]
                    llm_agent=get_langchain_agent(llm,template)
                    summary_response = summarize_html_report(vis_table,llm_agent)
                    
                except Exception as e:
                    if "openai" in str(e).lower() or "key" in str(e).lower() or "quota" in str(e).lower():
                        return jsonify({"require_api_key": True})  # 🔁 Trigger prompt in chat.js
                
                    summary_response = f"(⚠️ Could not summarize the plot: {e})"
            
                chat_history.append({"sender": "bot", "text": summary_response})
                #experimental resetting provider after use
                del session["valid_providers"]
                session["last_action"] = "summarized"
                session["chat_history"] = chat_history
                return jsonify({"chat_history": chat_history[-1:]})
        elif session.get("llm_provider") == "in-progress" and len(session.get("valid_providers"))==0:
            chat_history.append({
            "sender": "bot",
            "text": "You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated. To start again with another Data/ Model, click on 'reset chat' 🙂"
        })
            return jsonify({"chat_history": chat_history[-1:]})
    
    if not user_args and user_msg.lower() == "default":
        for arg in default_args:
            if arg not in user_args:
                user_args[arg]=default_args[arg]
        session["classifier"]=user_args["classifier"]
        chat_history.append({"sender": "bot", "text": "Running MMM_Fair with default parameters..."})
        vis_all, vis_fair = run_mmm_fair_app(default_args)
        
        default_resp="✅ Training complete! Here in the generated pareto plots (upper box) you see various solution points (models) available, where each solution point (denoted by theta: number) shows a different trade-off point between the different training objectives. \n\n The performance of the suggested best model is also plotted (lower box). You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated."
        session["last_action"] = "plotted"

        return jsonify({
                                "chat_history": [{
                                    "sender": "bot",
                                    "text": f"{default_resp}\n\n If you'd like me to explain what you're seeing, just type 'yes'.\n"
                                }],
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
            default_resp="✅ Training complete! Here in the generated pareto plots (upper box) you see various solution points (models) available, where each solution point (denoted by theta: number) shows a different trade-off point between the different training objectives. \n\n The performance of the suggested best model is also plotted (lower box). You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated."
            session["last_action"] = "plotted"
            
            return jsonify({
                            "chat_history": [{
                                "sender": "bot",
                                "text": f"{default_resp}\n\n If you'd like me to explain what you're seeing, just type 'yes'.\n"
                            }],
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
                chat_history.append({"sender": "bot", "text": f"That input wasn’t quite right: {err_msg}. Could you try entering {current_arg} again?"})
            else:
                user_args[current_arg] = clean_val
                chat_history.append({"sender": "bot", "text": f"Set {current_arg} = {clean_val}."})
        new_missing = get_missing_args(user_args)
        if new_missing:
            next_arg = new_missing[0]
            prompt = get_prompt_for_arg(next_arg, user_args)
            chat_history.append({"sender": "bot", "text": f"Thanks! Now let's handle the next step- \n{prompt}"})
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
        if c not in ["DP", "EP", "EO", "TPR", "FPR"]:
            return False, None, "Constraint must be DP, EP, EO, TPR, or FPR."
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
            k=''
            for prot in user_input.split(" "):
                if prot.lower() not in available_columns:
                    chosen_dataset = user_args.get("dataset", "").lower()
                    if chosen_dataset not in ['adult', 'bank']:
                        return False, None, f"Invalid protected. Available list of attributes in the data: {available_columns}."
                    else:
                        k+=' '+prot+','
                
            if k!='':
                return True, user_input.split(), f"Invalid protected attribute(s) {k} will be replaced with default known protected."
            
            return True, user_input.split(), ""

    elif arg_name == "nprotgs":
        existing_prots = user_args.get("prots", [])
        splitted = user_input.split()
        if len(splitted) != len(existing_prots):
            return False, None, f"Got {len(splitted)} non-protected vals for {len(existing_prots)} protected columns. Please match count."
        else:
            chosen_dataset = user_args.get("dataset", "").lower()
            if chosen_dataset not in ['adult', 'bank']:
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
    for fname in os.listdir(PLOT_DIR):
        if fname.startswith("table_") or fname.startswith("fair_") or fname.startswith("all_"):
            try:
                os.remove(os.path.join(PLOT_DIR, fname))
            except Exception:
                pass
    return "Chat reset done."

def run_mmm_fair_app(user_args):
    args = argparse.Namespace(**user_args)
    mmm_classifier, X_test, y_test, saIndex_test, sensitives=train(args)
    session["mmm_classifier"] = mmm_classifier
    session["xtest"]=X_test
    session["ytest"]=y_test
    session["saIndex_test"]=saIndex_test
    session["sensitives"]=sensitives
    PF=np.array([mmm_classifier.ob[i] for i in range(len(mmm_classifier.ob))])
    thetas=np.arange(len(mmm_classifier.ob))
    title=f"3D Scatter Plot. Showing various trade-off points between Accuracy, Balanced Accuracy, and Maximum violation of {mmm_classifier.constraint} fairness among protected attributes."
    vis_all=plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria="Multi",
           axis_names=['Acc.','Balanc. Acc', 'MMM-fair'],title=title, html=True)
    PF=np.array([mmm_classifier.fairobs[i] for i in range(len(mmm_classifier.fairobs))])
    title=f"3D Scatter Plot. Showing various trade-off points between maximum violation of Demopgraphic Parity, Equal Opportunity, and Equalized odds fairness for the given set of protected attributes."
    vis_fair=plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria= "Multi-definitions",
           axis_names=['DP','EqOpp', 'EqOdd'],title=title,html=True)

    y_pred= mmm_classifier.predict(X_test)
    report_table= generate_reports(
                                    'html', 
                                    sensitives, 
                                    mmm_classifier, 
                                    saIndex_test, 
                                    y_pred, 
                                    y_test,
                                    html=True
                                )
    plot_all = f"all_.html"
    plot_fair = f"fair_.html"
    plot_table = f"table_.html"

    plot_all_path = os.path.join(PLOT_DIR, plot_all)
    plot_fair_path = os.path.join(PLOT_DIR, plot_fair)
    plot_table_path = os.path.join(PLOT_DIR, plot_table)

    session['plot_table_path']=plot_table_path
    # Save the Plotly-generated HTML directly to files
    with open(plot_all_path, "w") as f:
        f.write(vis_all)
    with open(plot_fair_path, "w") as f:
        f.write(vis_fair)
    with open(plot_table_path, "w") as f:
        f.write(report_table)

    session["plot_all_url"]=f"/static/{plot_all}"
    session["plot_fair_url"]=f"/static/{plot_table}"
    return f"/static/{plot_all}", f"/static/{plot_table}"



@app.route('/static/<path:filename>')
def serve_plot(filename):
    plot_dir = os.path.abspath("static/plots")  # Ensure absolute path
    return send_from_directory(plot_dir, filename)   


@app.route("/update_model", methods=["POST"])
def update_model():
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

    X_test = session.get("xtest")
    y_test = session.get("ytest")
    user_args = session.get("user_args", {})
    sensitives = session.get("sensitives")
    saIndex_test = session.get("saIndex_test")


    y_pred = mmm_classifier.predict(X_test)

    report_table = generate_reports(
        'html',
        sensitives,
        mmm_classifier,
        saIndex_test,
        y_pred,
        y_test,
        html=True
    )

    unique_id = str(uuid.uuid4())[:4]
    plot_table = f"table_.html"
    p2=f"table_{unique_id}.html"
    plot_table_path = os.path.join(PLOT_DIR, p2)#plot_table)
    session['plot_table_path']=plot_table_path

    with open(plot_table_path, "w") as f:
        f.write(report_table)
        f.flush()
        os.fsync(f.fileno())


    timeout = 5  # max wait 5 seconds
    waited = 0
    while not os.path.exists(plot_table_path) and waited < timeout:
        time.sleep(0.1)
        waited += 0.1
        
    session["plot_fair_url"]=f"/static/{p2}"
    return jsonify({
        "success": True,
        "message": f"Model updated with Theta index {theta_index}.",
        "plot_fair_url": f"/static/{p2}?t={int(time.time())}"
    })

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
        user_args["save_path"] = save_path   # Update args with the user-selected path
        convert_to_onnx( mmm_classifier, save_path, xdata, clf)
        return jsonify({"success": True, "message": f"Model saved in {save_path}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()

